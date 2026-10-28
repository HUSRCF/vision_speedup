#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <unistd.h>

/* Opt-in storage adapter for NCCL2.20's hardcoded /dev/shm/nccl-XXXXXX.
 * Keep virtual names/suffixes unchanged, backing only this process tree's
 * new NCCL files by an owned, private tmpfs directory. No CUDA/NCCL symbols,
 * buffers, arithmetic, flags or timers are intercepted. */
static const char prefix[] = "/dev/shm/nccl-";
static char private_root[PATH_MAX];
static int enabled;
static int (*next_open)(const char *, int, ...);
static int (*next_open64)(const char *, int, ...);
static int (*next_unlink)(const char *);
static int (*next_mkstemp)(char *);
static int (*next_mkstemp64)(char *);

__attribute__((constructor)) static void initialize(void) {
    next_open = dlsym(RTLD_NEXT, "open");
    next_open64 = dlsym(RTLD_NEXT, "open64");
    next_unlink = dlsym(RTLD_NEXT, "unlink");
    next_mkstemp = dlsym(RTLD_NEXT, "mkstemp");
    next_mkstemp64 = dlsym(RTLD_NEXT, "mkstemp64");
    if (!next_open || !next_open64 || !next_unlink || !next_mkstemp || !next_mkstemp64) _exit(125);
    const char *root = getenv("TEAM07_NCCL_SHM_ROOT");
    if (!root || !*root) return;
    struct stat st;
    struct statfs fs;
    char resolved[PATH_MAX];
    if (root[0] != '/' || !realpath(root, resolved) || strcmp(root, resolved) ||
        lstat(root, &st) || !S_ISDIR(st.st_mode) || st.st_uid != getuid() ||
        (st.st_mode & 0777) != 0700 || statfs(root, &fs) || fs.f_type != 0x01021994 ||
        strlen(root) + 32 >= sizeof(private_root)) {
        static const char message[] = "NCCL_PRIVATE_TMPFS invalid private tmpfs root\n";
        ssize_t written = write(STDERR_FILENO, message, sizeof(message)-1);
        (void)written;
        _exit(125);
    }
    strcpy(private_root, root);
    enabled = 1;
}

static int translate(const char *name, char *out) {
    if (!enabled || !name || strncmp(name, prefix, sizeof(prefix)-1)) return 0;
    const char *suffix = name + sizeof(prefix)-1;
    if (strlen(suffix) != 6) return 0;
    for (int i=0; i<6; ++i) {
        char c=suffix[i];
        if (!((c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9'))) return 0;
    }
    size_t length=strlen(private_root);
    memcpy(out, private_root, length);
    memcpy(out+length, "/nccl-", 6);
    memcpy(out+length+6, suffix, 7);
    return 1;
}

static int temporary(char *name, int large) {
    char actual[PATH_MAX];
    int redirected = translate(name, actual);
    int result = large ? next_mkstemp64(redirected ? actual : name) : next_mkstemp(redirected ? actual : name);
    if (redirected && result >= 0) memcpy(name+sizeof(prefix)-1, actual+strlen(actual)-6, 6);
    return result;
}
int mkstemp(char *name) { return temporary(name, 0); }
int mkstemp64(char *name) { return temporary(name, 1); }

static int needs_mode(int flags) { return (flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE); }
int open(const char *name, int flags, ...) {
    mode_t mode=0;
    if (needs_mode(flags)) { va_list args; va_start(args, flags); mode=va_arg(args, int); va_end(args); }
    char actual[PATH_MAX];
    return next_open(translate(name, actual) ? actual : name, flags, mode);
}
int open64(const char *name, int flags, ...) {
    mode_t mode=0;
    if (needs_mode(flags)) { va_list args; va_start(args, flags); mode=va_arg(args, int); va_end(args); }
    char actual[PATH_MAX];
    return next_open64(translate(name, actual) ? actual : name, flags, mode);
}
int __open_2(const char *name, int flags) {
    if (needs_mode(flags)) { errno=EINVAL; return -1; }
    return open(name, flags);
}
int __open64_2(const char *name, int flags) {
    if (needs_mode(flags)) { errno=EINVAL; return -1; }
    return open64(name, flags);
}
int unlink(const char *name) {
    char actual[PATH_MAX];
    return next_unlink(translate(name, actual) ? actual : name);
}
