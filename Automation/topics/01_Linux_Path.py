"""
Linux — The Open-Source Operating System That Runs the World
=============================================================

Linux is the kernel at the heart of the most widely deployed operating
system in history. It runs on Android phones, cloud servers (96% of the
top million web servers), supercomputers (100% of the Top500), embedded
systems, IoT devices, Chromebooks, Smart TVs, and increasingly the desktop.

Created by Linus Torvalds in 1991 as a free alternative to MINIX for his
Intel 386 PC, Linux grew from a hobby project into the backbone of global
digital infrastructure — powering Google, Amazon, Facebook, and every major
AI training cluster in existence.

What makes Linux different from Windows and macOS:
    - Open source: the kernel source is readable, modifiable, distributable
    - Multi-user: designed from day one for many users simultaneously
    - Modular: swap out any component — the shell, the display server, the init system
    - Stable: servers run for years without rebooting
    - Free: no licensing cost — the entire OS stack is yours

Understanding Linux means understanding how modern computing actually works —
from how files and permissions are stored on disk, to how processes are
scheduled, to how the network stack processes packets, to how the shell
interprets every command you type.

This module covers: Linux distributions, the filesystem hierarchy, processes,
users and permissions, the shell and scripting, networking, package management,
and a comprehensive command reference for daily use.

"""

import textwrap
import re
import importlib.util
from pathlib import Path

TOPIC_NAME = "Linux — The Open-Source Operating System"
DISPLAY_NAME = "00 · Linux"
ICON = "🐧"
SUBTITLE = "Distributions, Filesystem, Processes, Shell and Command Mastery"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = r"""


### PART 1 — LINUX DISTRIBUTIONS: ONE KERNEL, MANY FLAVOURS

### What a Distribution Is

    The Linux KERNEL is just the core operating system component — it manages
    hardware, memory, processes, and files. A Linux DISTRIBUTION (or "distro")
    bundles the kernel with:
        - A package manager (how you install software)
        - An init system (how services start)
        - System utilities (GNU coreutils, bash, etc.)
        - A default desktop environment (optional)
        - Repository of pre-compiled software packages
        - A release schedule and support policy

    There are 600+ active Linux distributions. Most trace ancestry to one of
    three major families.

### The Three Major Families

    DEBIAN FAMILY (largest ecosystem):
        Debian:       The "universal OS". Rock-solid, slow release cycle.
                      Basis for many other distros.
        Ubuntu:       Most popular desktop Linux. 6-month releases.
                      LTS versions (e.g., 22.04) supported 5 years.
        Ubuntu variants: Kubuntu (KDE), Lubuntu (LXQt), Xubuntu (Xfce)
        Linux Mint:   Ubuntu-based, very beginner-friendly
        Kali Linux:   Security/penetration testing distro (Debian-based)
        Raspberry Pi OS: Debian-based for ARM (Raspberry Pi)
        Pop!_OS:      System76's Ubuntu-based distro (good for ML/gaming)
        Package manager: APT (Advanced Package Tool)  →  apt, apt-get, dpkg

    RED HAT FAMILY (enterprise market):
        RHEL:         Red Hat Enterprise Linux. Commercial, paid support.
                      The enterprise standard — used in most Fortune 500.
        CentOS:       Was the free RHEL rebuild. Now CentOS Stream (rolling).
        AlmaLinux:    Free RHEL rebuild replacing CentOS (1:1 compatible)
        Rocky Linux:  Another free RHEL rebuild (founded by CentOS creator)
        Fedora:       Red Hat's upstream sandbox. Cutting-edge, 6-month releases.
        Oracle Linux: Oracle's RHEL rebuild (with Oracle kernel patches)
        Amazon Linux: AWS's RHEL-based distro (for EC2 instances)
        Package manager: YUM → DNF  →  dnf, yum, rpm

    SUSE FAMILY (European enterprise):
        SLES:         SUSE Linux Enterprise Server. SAP-certified, 10-yr support.
        openSUSE Leap:  Free SLES rebuild. Stable, point releases.
        openSUSE Tumbleweed: Rolling release, always latest packages.
        Package manager: Zypper  →  zypper, rpm

    INDEPENDENT (notable):
        Arch Linux:   Rolling release, minimal base, highly customisable.
                      Famous for: PKGBUILD, AUR (Arch User Repository)
                      Philosophy: install only what you need
        Gentoo:       Source-based — everything compiled from source
                      for your exact hardware (USE flags)
        NixOS:        Declarative config — entire OS state in one config file
                      Reproducible builds, atomic rollbacks
        Alpine Linux: Tiny (5 MB), musl libc instead of glibc
                      Dominant in Docker containers (most Docker base images)
        Package manager: pacman (Arch), emerge (Gentoo), nix (Nix), apk (Alpine)

### Distribution Selection Guide

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Use Case                    │ Recommended Distro                     │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Beginner desktop            │ Ubuntu 22.04 LTS or Linux Mint         │
    │ ML/AI workstation           │ Ubuntu 22.04 LTS or Pop!_OS            │
    │ Enterprise server           │ RHEL / AlmaLinux / Rocky Linux         │
    │ AWS cloud instances         │ Amazon Linux 2023 or Ubuntu 22.04      │
    │ Google Cloud                │ Debian 11/12 or Ubuntu 22.04           │
    │ Azure                       │ Ubuntu 22.04 LTS (Microsoft's default) │
    │ Docker containers           │ Alpine Linux (minimal) or Debian slim  │
    │ Bleeding-edge desktop       │ Fedora or Arch Linux                   │
    │ Security / pen testing      │ Kali Linux                             │
    │ Raspberry Pi / ARM          │ Raspberry Pi OS or Ubuntu 22.04 ARM    │
    │ Reproducible infra          │ NixOS                                  │
    │ Learning Linux internals    │ Arch Linux (you build it yourself)     │
    └──────────────────────────────────────────────────────────────────────┘

### Version Naming Conventions

    Ubuntu: YY.MM (22.04 = April 2022, 24.04 = April 2024)
    LTS versions: April of even years (20.04, 22.04, 24.04) — 5-year support
    Non-LTS: every 6 months — 9-month support

    RHEL: major.minor (RHEL 9.2). Major versions: 5–10 years support.
    CentOS/AlmaLinux/Rocky: mirrors RHEL version numbers exactly.

    Fedora: numeric (F38, F39, F40) — 1 release/year, 13-month support.

    openSUSE Leap: mirrors SLES (15.5, 15.6)
    openSUSE Tumbleweed: no version — always latest (rolling)


### PART 2 — THE LINUX KERNEL AND ARCHITECTURE

### The Kernel's Responsibilities

    The Linux kernel sits between hardware and user programs:

    ┌────────────────────────────────────────────────────────────────────┐
    │  User space:  your applications, shell, Python, web browser        │
    ├────────────────────────────────────────────────────────────────────┤
    │  System call interface (the contract between user and kernel)      │
    ├────────────────────────────────────────────────────────────────────┤
    │  Linux Kernel:                                                     │
    │    Process scheduler  │  Memory manager  │  VFS (filesystem)       │
    │    Network stack      │  Device drivers  │  IPC mechanisms         │
    ├────────────────────────────────────────────────────────────────────┤
    │  Hardware: CPU, RAM, Disk, NIC, GPU, USB, ...                      │
    └────────────────────────────────────────────────────────────────────┘

    Key kernel subsystems:
        Process scheduler:  decides which process runs on which CPU core.
                            Linux uses CFS (Completely Fair Scheduler).
        Memory manager:     virtual memory, page tables, swapping, OOM killer.
        VFS (Virtual Filesystem):  abstract filesystem layer — same API
                            whether the FS is ext4, XFS, Btrfs, NTFS, NFS.
        Network stack:      TCP/IP, UDP, sockets, netfilter (firewall).
        Device drivers:     kernel modules (.ko files) for hardware.
        IPC:                pipes, signals, sockets, shared memory, semaphores.

### Kernel Space vs User Space

    KERNEL SPACE: the kernel runs here. Direct hardware access.
    Full trust — a bug can crash the system.

    USER SPACE: all applications run here. No direct hardware access.
    Isolated — a buggy app crashes itself, not the OS.
    Applications talk to the kernel via SYSTEM CALLS.

    System call examples:
        open()    → kernel opens a file, returns a file descriptor
        read()    → kernel reads bytes from a file descriptor
        write()   → kernel writes bytes to a file descriptor
        fork()    → kernel creates a copy of the calling process
        exec()    → kernel replaces current process with a new program
        mmap()    → kernel maps a file or memory into process address space
        socket()  → kernel creates a network socket
        kill()    → kernel sends a signal to a process

    When you type ls in the terminal:
        bash → fork() to create child process → exec("ls") → ls opens /proc
        ls calls readdir() system calls → kernel reads directory entries
        ls writes to stdout → kernel write() → terminal emulator displays

### Monolithic vs Microkernel

    Linux is a MONOLITHIC kernel with loadable modules:
        All core OS services (scheduler, VFS, networking) run in kernel space.
        Device drivers are loadable modules (added/removed at runtime).
        Fast: no context switches between kernel components.

    Microkernel (macOS XNU, QNX, Minix) runs services in user space.
    Slower due to IPC overhead, but more robust isolation.

    Why Linux won: monolithic = faster; modules = flexibility without reboot.


### PART 3 — THE FILESYSTEM HIERARCHY STANDARD (FHS)

### Everything is a File

    Linux's most fundamental principle: EVERYTHING IS A FILE.
        Regular files:     /home/alice/document.txt
        Directories:       /home/alice/projects/ (a file listing other files)
        Devices:           /dev/sda (your hard disk — read/write like a file)
        Sockets:           /var/run/docker.sock (IPC via file-like interface)
        Pipes:             /proc/self/fd/0 (standard input is a file)
        Symlinks:          /usr/bin/python → /usr/bin/python3.11
        Pseudo-files:      /proc/cpuinfo (reads kernel data structures)

    This uniformity means: one API (open/read/write/close) works for
    files, network connections, hardware devices, and inter-process comms.

### The Directory Tree

    / ─────────────────── ROOT of the filesystem (not /root — that's home!)
    ├── bin  ───────────── Essential user binaries (ls, cp, cat, bash)
    │                      (in modern systems: symlink to /usr/bin)
    ├── sbin ───────────── System binaries for root (ifconfig, fdisk, init)
    │                      (in modern systems: symlink to /usr/sbin)
    ├── usr ────────────── User programs and data (the big one)
    │   ├── bin          ← most user commands live here
    │   ├── sbin         ← system administration commands
    │   ├── lib          ← shared libraries (.so files)
    │   ├── local        ← locally compiled software (outside package mgr)
    │   │   ├── bin, lib, include, share
    │   ├── share        ← architecture-independent data (man pages, icons)
    │   └── include      ← C/C++ header files (gcc uses these)
    ├── etc ────────────── System configuration files (text-based)
    │   ├── passwd       ← user account database
    │   ├── shadow       ← hashed passwords (root-only readable)
    │   ├── hosts        ← static hostname↔IP mappings
    │   ├── fstab        ← filesystem mount table
    │   ├── sudoers      ← who can run sudo and what they can run
    │   ├── apt/         ← APT configuration (Debian/Ubuntu)
    │   ├── yum.repos.d/ ← YUM repository definitions (RHEL/Fedora)
    │   ├── systemd/     ← systemd service unit files
    │   └── ssh/         ← SSH server configuration
    ├── home ──────────── User home directories
    │   ├── alice/       ← alice's home: /home/alice
    │   └── bob/
    ├── root ──────────── Root user's home directory (separate from /home!)
    ├── var ────────────── Variable data (changes during operation)
    │   ├── log/         ← system and application log files
    │   ├── lib/         ← persistent application state
    │   ├── cache/       ← application cache data
    │   ├── run/         ← runtime data (PID files, sockets)
    │   └── spool/       ← queued data (print jobs, mail)
    ├── tmp ────────────── Temporary files (cleared on reboot)
    ├── proc ──────────── Virtual FS: kernel and process information
    │   ├── cpuinfo      ← CPU details (cores, MHz, flags)
    │   ├── meminfo      ← RAM usage statistics
    │   ├── loadavg      ← system load averages
    │   ├── 1234/        ← directory for process with PID 1234
    │   │   ├── status   ← process status
    │   │   ├── maps     ← memory map
    │   │   ├── fd/      ← open file descriptors
    │   │   └── cmdline  ← command that started the process
    │   └── net/         ← network interface and routing info
    ├── sys ────────────── Virtual FS: kernel device and driver information
    │   └── class/net/   ← network device configuration
    ├── dev ────────────── Device files
    │   ├── sda          ← first SATA/SAS disk (b=block device)
    │   ├── sda1         ← first partition of sda
    │   ├── nvme0n1      ← first NVMe SSD
    │   ├── tty          ← terminal devices
    │   ├── null         ← the void (writes discarded, reads return EOF)
    │   ├── zero         ← infinite stream of null bytes
    │   ├── random       ← cryptographically random bytes
    │   └── urandom      ← faster but slightly less random bytes
    ├── mnt ────────────── Temporary mount points (mount a disk here)
    ├── media ─────────── Auto-mounted removable media (USB drives, DVDs)
    ├── lib ────────────── Shared libraries for /bin and /sbin binaries
    ├── lib64 ─────────── 64-bit shared libraries
    ├── boot ──────────── Bootloader files (kernel image, initrd)
    │   ├── vmlinuz      ← compressed kernel image
    │   ├── initrd.img   ← initial RAM disk (early boot filesystem)
    │   └── grub/        ← GRUB bootloader configuration
    └── opt ────────────── Optional/third-party application packages

### Filesystem Types

    ext4:     Default on Debian/Ubuntu/RHEL. Journaled, mature, stable.
    XFS:      Default on RHEL/Fedora. Excellent for large files, parallel I/O.
    Btrfs:    Copy-on-write, snapshots, built-in RAID, checksumming.
              Default on openSUSE and Fedora (post-F33).
    ZFS:      Advanced (snapshots, RAID-Z, dedup) — not in mainline kernel.
              Available via OpenZFS project.
    tmpfs:    RAM-based FS — fast, lost on reboot. Used for /tmp, /run.
    procfs:   /proc — virtual FS exposing kernel internals.
    sysfs:    /sys — virtual FS for device/driver configuration.
    ext2/3:  Older; ext3 added journaling over ext2; ext4 supersedes both.
    NTFS/vFAT: Windows filesystems — readable (and writable) on Linux.
    NFS:      Network File System — mount remote directories over network.
    CIFS/SMB: Windows network shares — mount via mount.cifs.


### PART 4 — USERS, GROUPS AND PERMISSIONS

### User Model

    Every Linux user has:
        UID (User ID):     unique integer. Root = 0. System = 1–999. Regular = 1000+
        GID (Group ID):    primary group. Secondary groups also assignable.
        Home directory:    /home/username (root = /root)
        Login shell:       /bin/bash, /bin/zsh, /usr/sbin/nologin (no login)

    Stored in:
        /etc/passwd:   username:x:UID:GID:comment:home:shell
        /etc/shadow:   username:hashed_password:last_change:...
        /etc/group:    group_name:x:GID:member1,member2

    Special users:
        root (UID 0):    superuser — can do anything. No permission checks.
        nobody:          unprivileged user for daemon isolation
        www-data:        web server user (Apache/Nginx)
        systemd-*:       various systemd service users
        _apt, syslog:    system service users

### The Permission Model

    Every file/directory has three permission triads:
        Owner (u):   the user who owns the file
        Group (g):   the group that owns the file
        Others (o):  everyone else

    Each triad has three bits:
        r (read):    4  → can read file contents / list directory
        w (write):   2  → can modify file / create-delete in directory
        x (execute): 1  → can run as program / enter directory (cd)

    Displayed as:    -rwxr-xr--
                     │└─┬─┘└─┬─┘└─┬─┘
                     │  owner group others
                     └── file type: - regular, d directory, l symlink,
                                    b block device, c char device, s socket

    Examples:
        -rw-r--r--   644  Regular file: owner read+write, others read-only
        -rwxr-xr-x   755  Executable: owner all, group+others read+execute
        drwxr-xr-x   755  Directory: owner all, others can list and enter
        drwx------   700  Private directory: owner only
        -rw-------   600  Private file: owner only (e.g., ~/.ssh/id_rsa)
        -rwsr-xr-x   4755 SUID bit: runs as owner (e.g., /usr/bin/passwd)
        -rwxrwxrwx   777  Everyone has all permissions (DANGEROUS)

### Special Permission Bits

    SUID (Set User ID) bit — 4000:
        On executable: runs as the FILE OWNER, not the caller.
        Example: /usr/bin/passwd (SUID root) — ordinary user can change
        their own password because passwd temporarily runs as root.
        ls shows: -rwsr-xr-x   (s in owner execute position)

    SGID (Set Group ID) bit — 2000:
        On executable: runs as the FILE GROUP.
        On directory: new files inherit the directory's group.
        ls shows: -rwxr-sr-x   (s in group execute position)

    Sticky bit — 1000:
        On directory: users can only delete their OWN files.
        Used on /tmp: everyone can write, but can't delete others' files.
        ls shows: drwxrwxrwt   (t in others execute position)

### sudo — Privilege Escalation

    sudo (superuser do) allows specific users to run specific commands as root:
        sudo apt update          → run apt update as root
        sudo -i                  → open a root shell (interactive)
        sudo -u www-data cmd     → run cmd as user www-data

    /etc/sudoers (edit with visudo — validates syntax before saving):
        alice  ALL=(ALL:ALL) ALL    → alice can sudo anything
        bob    ALL=(ALL) NOPASSWD: /usr/bin/apt  → bob can run apt without password
        %admin ALL=(ALL) ALL        → all users in 'admin' group can sudo

    Ubuntu: first user automatically gets sudo access (member of sudo group).
    RHEL/Fedora: first user in wheel group gets sudo access.


### PART 5 — PROCESSES AND JOB CONTROL

### Process Fundamentals

    A process is a running program with its own:
        PID (Process ID):    unique integer identifier
        PPID (Parent PID):   the process that created it
        UID/GID:             which user it runs as
        Address space:       private virtual memory region
        File descriptors:    open files, sockets, pipes
        Working directory:   the "current directory" of the process
        Environment variables: key=value pairs inherited from parent

    Process states:
        R  Running / runnable (on CPU or waiting for CPU)
        S  Interruptible sleep (waiting for I/O, can be woken by signal)
        D  Uninterruptible sleep (waiting for disk I/O — can't be killed!)
        T  Stopped (paused by Ctrl+Z or SIGSTOP)
        Z  Zombie (exited but parent hasn't collected exit status)
        I  Idle kernel thread

### The Process Tree

    PID 1 is the init system (systemd on modern Linux).
    ALL other processes are descendants of PID 1.

    PID 1 (systemd)
    └── getty (login prompt on tty1)
        └── bash (your shell after login)
            └── vim (you opened an editor)
    └── sshd (SSH server daemon)
        └── sshd (your SSH connection)
            └── bash
                └── python train.py  ← your ML job
    └── nginx (web server)
    └── dockerd (Docker daemon)
        └── containerd
            └── containers...

### Signals — Process Communication

    Signals are software interrupts sent to processes:

    SIGHUP  (1):   Hang up. Conventionally: reload config (nginx, sshd).
    SIGINT  (2):   Interrupt. Ctrl+C → sent to foreground process.
    SIGQUIT (3):   Quit. Ctrl+\  → generates a core dump.
    SIGKILL (9):   Kill. CANNOT be caught or ignored. Immediate death.
                   Last resort — may leave resources unfreed.
    SIGTERM (15):  Terminate. The polite kill. Process can catch and clean up.
    SIGSTOP (19):  Stop/pause process. CANNOT be caught or ignored.
    SIGCONT (18):  Continue a stopped process.
    SIGUSR1 (10):  User-defined signal 1 (app-specific meaning)
    SIGUSR2 (12):  User-defined signal 2 (app-specific meaning)
    SIGCHLD (17):  Child process changed state (parent is notified)
    SIGSEGV (11):  Segmentation fault (invalid memory access — app bug)
    SIGPIPE (13):  Broken pipe (write to pipe with no reader)
    SIGALRM (14):  Alarm clock (timer expired)

    Sending signals:
        kill -15 1234      → send SIGTERM to PID 1234 (polite)
        kill -9 1234       → send SIGKILL to PID 1234 (force)
        killall nginx      → send SIGTERM to all processes named "nginx"
        pkill -f "python"  → kill all processes matching "python" in cmdline
        Ctrl+C             → SIGINT to foreground process
        Ctrl+Z             → SIGSTOP to foreground process

### Job Control

    Run in background:   command &       → shell returns immediately
    List jobs:           jobs            → [1]+ Running  command &
    Bring to foreground: fg %1           → resumes job 1 in foreground
    Send to background:  bg %1           → resumes stopped job in background
    Stop foreground:     Ctrl+Z          → pause current command
    Disown a job:        disown %1       → job survives shell exit

### Daemons and Services

    A daemon is a background process with no controlling terminal:
        - Typically started at boot by the init system
        - Named with trailing 'd': sshd, httpd, crond, systemd
        - Runs as a dedicated non-root user for security isolation

    systemd manages services:
        systemctl start nginx          → start now
        systemctl stop nginx           → stop now
        systemctl restart nginx        → stop + start
        systemctl reload nginx         → reload config without restart
        systemctl enable nginx         → auto-start at boot
        systemctl disable nginx        → don't auto-start at boot
        systemctl status nginx         → show status + recent logs
        systemctl is-active nginx      → prints "active" or "inactive"
        journalctl -u nginx            → show all logs for nginx
        journalctl -u nginx -f         → follow nginx logs live


    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PART 6 — THE SHELL: BASH AND THE COMMAND LINE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### What the Shell Is

    The shell is a command interpreter — it reads commands you type,
    executes programs, and displays output.

    Common shells:
        bash:   Bourne Again SHell. Default on most Linux distros.
        zsh:    Z Shell. Default on macOS. Oh-My-Zsh ecosystem. Powerful.
        sh:     POSIX-compliant minimal shell. Used in scripts for portability.
        fish:   Friendly interactive shell. Great UX, non-POSIX.
        dash:   Lightweight POSIX shell. /bin/sh on Ubuntu (fast scripts).
        ksh:    KornShell. Popular in enterprise UNIX environments.

### Shell Mechanics

    The shell executes a command by:
        1. Parsing the command line (tokenise, handle quotes)
        2. Expanding: variables ($VAR), globs (*.txt), tilde (~)
        3. Forking a child process
        4. Setting up I/O (stdin/stdout/stderr, pipes, redirects)
        5. exec()-ing the program into the child process
        6. Waiting for the child to exit (unless & is used)
        7. Collecting the exit code ($?)

### I/O Redirection and Pipes

    Standard streams:
        stdin  (fd 0):  input — keyboard by default
        stdout (fd 1):  output — terminal by default
        stderr (fd 2):  error output — terminal by default

    Redirecting output:
        command > file          → redirect stdout to file (overwrite)
        command >> file         → redirect stdout to file (append)
        command 2> file         → redirect stderr to file
        command 2>&1            → redirect stderr to same place as stdout
        command > out 2> err    → stdout to out, stderr to err
        command &> file         → redirect both stdout and stderr (bash)
        command > /dev/null 2>&1 → discard ALL output

    Redirecting input:
        command < file          → read stdin from file
        command << EOF          → here-doc: read until EOF marker
        command <<< "string"    → here-string: feed string as stdin

    Pipes (connect stdout of one to stdin of next):
        cmd1 | cmd2             → pipe cmd1 output to cmd2 input
        cmd1 | cmd2 | cmd3      → pipeline chain
        cmd1 |& cmd2            → pipe both stdout AND stderr (bash 4+)

    Process substitution (treat command output as a file):
        diff <(sort file1) <(sort file2)   → compare sorted versions
        cat <(echo "header") file.txt       → prepend header

### Variables and Environment

    Shell variables:
        VAR="value"             → assign (no spaces around =)
        echo $VAR               → use the variable
        echo "${VAR}"           → safer: explicit boundary
        unset VAR               → delete variable
        readonly VAR="value"    → make immutable

    Environment variables (exported to child processes):
        export VAR="value"      → make available to all child processes
        env                     → list all environment variables
        printenv VAR            → print one variable

    Special variables:
        $0      → script name
        $1, $2  → positional arguments
        $@      → all arguments (as separate words)
        $#      → number of arguments
        $?      → exit code of last command (0 = success, non-zero = error)
        $$      → PID of current shell
        $!      → PID of last background process
        $_      → last argument of previous command

    Important environment variables:
        PATH    → colon-separated list of directories searched for commands
        HOME    → current user's home directory
        USER    → current username
        SHELL   → path to current shell (/bin/bash)
        EDITOR  → default text editor (nano, vim, emacs)
        LANG    → locale (en_US.UTF-8)
        TERM    → terminal type (xterm-256color)
        PS1     → primary prompt string

### Bash Configuration Files

    Login shell (ssh, su -, terminal login):
        /etc/profile      → system-wide (run once)
        ~/.bash_profile   → user-specific (run once)
        ~/.bashrc         → usually sourced from .bash_profile

    Interactive non-login shell (new terminal tab):
        /etc/bash.bashrc  → system-wide
        ~/.bashrc         → user-specific (run each new shell)

    ~/.bashrc is where you put:
        Aliases:    alias ll='ls -alFh'
        PATH:       export PATH="$HOME/.local/bin:$PATH"
        Prompt:     PS1='\u@\h:\w\$ '
        Functions:  function mkcd() { mkdir -p "$1" && cd "$1"; }

### Globbing (Filename Expansion)

    *       → any string of characters (excluding /)
              ls *.py        → all Python files
    ?       → exactly one character
              ls file?.txt   → file1.txt, fileA.txt, but not file10.txt
    [abc]   → one character from the set
              ls file[123].txt → file1.txt, file2.txt, file3.txt
    [a-z]   → range of characters
              ls [A-Z]*.txt  → .txt files starting with uppercase
    {a,b,c} → brace expansion (not a glob — expanded before execution)
              cp file.txt{,.bak} → copies file.txt to file.txt.bak
              echo {1..10}       → 1 2 3 4 5 6 7 8 9 10
    **      → recursive glob (bash 4+ with globstar option)
              ls **/*.py     → all Python files in all subdirectories


    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PART 7 — PACKAGE MANAGEMENT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### APT — Debian/Ubuntu

    apt (high-level, recommended):
        sudo apt update                      → refresh package lists from repos
        sudo apt upgrade                     → upgrade all installed packages
        sudo apt full-upgrade                → upgrade + remove obsoletes
        sudo apt install nginx               → install nginx and dependencies
        sudo apt install -y nginx python3    → install multiple, auto-confirm
        sudo apt remove nginx                → remove but keep config files
        sudo apt purge nginx                 → remove + delete config files
        sudo apt autoremove                  → remove unused dependencies
        sudo apt search keyword              → search package names/descriptions
        apt list --installed                 → list installed packages
        apt show nginx                       → show package details
        sudo apt install ./local.deb         → install a local .deb file

    dpkg (low-level, for .deb files):
        sudo dpkg -i package.deb            → install a .deb file
        sudo dpkg -r package                → remove package
        dpkg -l                             → list installed packages
        dpkg -L nginx                       → list files installed by nginx
        dpkg -S /usr/bin/nginx              → which package owns this file

    Repositories:
        /etc/apt/sources.list               → main repo list
        /etc/apt/sources.list.d/            → additional repo files
        sudo add-apt-repository ppa:name    → add a PPA (Ubuntu only)

### DNF/YUM — RHEL/Fedora/CentOS

    dnf (modern, default on RHEL 8+, Fedora):
        sudo dnf update                     → update all packages
        sudo dnf install nginx              → install
        sudo dnf remove nginx               → uninstall
        sudo dnf search keyword             → search
        sudo dnf info nginx                 → package details
        dnf list installed                  → list installed
        sudo dnf install ./package.rpm      → install local .rpm
        sudo dnf clean all                  → clean cached package data
        sudo dnf history                    → show transaction history
        sudo dnf history undo 5             → undo transaction #5 (rollback!)

    rpm (low-level):
        sudo rpm -i package.rpm             → install
        sudo rpm -e package                 → remove
        rpm -qa                             → list all installed packages
        rpm -ql nginx                       → list files from nginx package
        rpm -qf /usr/sbin/nginx             → which package owns this file

### Zypper — openSUSE/SLES

        sudo zypper refresh                 → refresh repos
        sudo zypper update                  → update all
        sudo zypper install nginx           → install
        sudo zypper remove nginx            → remove
        zypper search keyword               → search
        sudo zypper dup                     → distribution upgrade

### Universal Package Formats

    Snap (Canonical — sandboxed apps):
        sudo snap install vscode --classic  → install VS Code snap
        snap list                           → list installed snaps
        sudo snap remove vscode             → remove

    Flatpak (cross-distro sandboxed apps):
        flatpak install flathub app.id      → install from Flathub
        flatpak list                        → list installed
        flatpak run app.id                  → run a Flatpak app

    AppImage (portable executables):
        chmod +x app.AppImage
        ./app.AppImage                      → just run it (no install)


    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PART 8 — NETWORKING AND SECURITY ESSENTIALS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Network Configuration

    ip command (modern — replaces ifconfig, route):
        ip addr show                        → show all interfaces and IPs
        ip addr show eth0                   → show eth0 specifically
        ip link show                        → show interface link state
        ip link set eth0 up/down            → bring interface up/down
        ip route show                       → show routing table
        ip route add default via 192.168.1.1 → add default gateway
        ip neigh show                       → show ARP table (MAC ↔ IP)

    ifconfig (older — still common):
        ifconfig                            → show all interfaces
        ifconfig eth0 192.168.1.100/24      → set IP address

    Networking files:
        /etc/hostname               → machine's hostname
        /etc/hosts                  → static hostname → IP mappings
        /etc/resolv.conf            → DNS server addresses
        /etc/network/interfaces     → Debian network config
        /etc/netplan/*.yaml         → Ubuntu 17.10+ network config
        /etc/sysconfig/network-scripts/ → RHEL network config (older)

### SSH — Secure Shell

    Connecting:
        ssh user@hostname           → connect as user
        ssh -p 2222 user@host       → connect on non-standard port
        ssh -i ~/.ssh/key user@host → use specific private key
        ssh -L 8080:localhost:80 user@host → local port forwarding
        ssh -R 9090:localhost:3000 user@host → remote port forwarding
        ssh -J jump@bastion user@target → jump through bastion host

    Key management:
        ssh-keygen -t ed25519 -C "email"    → generate key pair (Ed25519, modern)
        ssh-keygen -t rsa -b 4096           → generate RSA 4096-bit key
        ssh-copy-id user@host               → copy public key to remote host
        ssh-add ~/.ssh/id_ed25519           → add key to SSH agent

    Config file (~/.ssh/config):
        Host myserver
            HostName 192.168.1.100
            User alice
            Port 2222
            IdentityFile ~/.ssh/id_ed25519
        # Then just: ssh myserver

    Server config (/etc/ssh/sshd_config):
        PasswordAuthentication no    → disable password login (use keys only)
        PermitRootLogin no           → disable direct root login
        Port 2222                    → use non-standard port

### Firewall — UFW and firewalld

    UFW (Uncomplicated Firewall — Ubuntu default):
        sudo ufw status verbose             → show rules
        sudo ufw enable / disable
        sudo ufw allow 22/tcp               → allow SSH
        sudo ufw allow 80,443/tcp           → allow HTTP + HTTPS
        sudo ufw deny 3306                  → block MySQL port
        sudo ufw allow from 192.168.1.0/24  → allow from subnet
        sudo ufw delete allow 80/tcp        → remove a rule

    firewalld (RHEL/Fedora default):
        sudo firewall-cmd --state           → running or not
        sudo firewall-cmd --list-all        → show current rules
        sudo firewall-cmd --add-service=http --permanent
        sudo firewall-cmd --add-port=8080/tcp --permanent
        sudo firewall-cmd --reload          → apply permanent rules

### System Logs

    systemd journal (modern — persistent, structured):
        journalctl                          → all logs
        journalctl -f                       → follow live
        journalctl -u nginx                 → logs for nginx service
        journalctl -u nginx --since today   → today's nginx logs
        journalctl -n 100                   → last 100 lines
        journalctl --since "2024-01-01" --until "2024-01-02"
        journalctl -p err                   → only error-level and above
        journalctl -b                       → since last boot
        journalctl -b -1                    → previous boot's logs

    Traditional log files (in /var/log/):
        /var/log/syslog             → general system messages (Debian)
        /var/log/messages           → general system messages (RHEL)
        /var/log/auth.log           → authentication (sudo, ssh logins)
        /var/log/secure             → authentication (RHEL)
        /var/log/kern.log           → kernel messages
        /var/log/apt/               → apt package management activity
        /var/log/nginx/             → nginx access + error logs
        /var/log/mysql/             → MySQL server logs
        tail -f /var/log/syslog     → follow syslog live

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "1. Essential Commands — Navigation, Files, and Text Processing": {

        "description": (
            "Comprehensive reference for daily Linux commands. "
            "Navigation and filesystem operations. "
            "File creation, copying, moving, deletion. "
            "Text viewing, searching, and processing (grep, sed, awk, sort). "
            "Archiving and compression. "
            "Finding files. All commands run on any Linux system."
        ),
        "runnable": True,
        "pipeline_cmd": "token",
        "code": r'''
import subprocess
import os
import tempfile

print("=" * 65)
print("  ESSENTIAL LINUX COMMANDS — NAVIGATION, FILES & TEXT")
print("=" * 65)
print()

def run(cmd, input_data=None, show_output=True):
    """Run a shell command and display the result."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, encoding="utf-8",
        input=input_data
    )
    output = result.stdout.strip() or result.stderr.strip()
    if show_output and output:
        for line in output.split('\n')[:15]:    # cap at 15 lines
            print(f"    {line}")
    return output

# Work in a temp directory so we don't pollute the system
DEMO = tempfile.mkdtemp(prefix="linux_demo_")
os.chdir(DEMO)
print(f"  Working in: {DEMO}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Navigation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Navigation commands")
print("━" * 65)
print()

commands_nav = """
  pwd                      Print current Working Directory
  cd /etc                  Change Directory (absolute path)
  cd ..                    Go up one level
  cd ~                     Go to home directory
  cd -                     Go to previous directory
  ls                       List files
  ls -l                    Long format (permissions, size, date)
  ls -la                   Long format + hidden files
  ls -lh                   Long format + human-readable sizes
  ls -lt                   Sort by modification time (newest first)
  ls -lS                   Sort by file size (largest first)
  ls -R                    Recursive listing
  tree                     Show directory tree (if installed)
  tree -L 2                Tree limited to 2 levels deep
  pushd /tmp               Go to /tmp and push current dir onto stack
  popd                     Go back to previous dir (pop stack)
  dirs                     Show directory stack
"""
print(commands_nav)

# Live demo
print("  Live demo:")
run("pwd")
os.makedirs("projects/ml/data", exist_ok=True)
os.makedirs("projects/web", exist_ok=True)
run("ls -la .")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: File and directory operations
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — File and directory operations")
print("━" * 65)
print()

commands_files = """
  touch file.txt           Create empty file / update timestamp
  mkdir -p a/b/c           Make directories (including parents)
  cp file1 file2           Copy file
  cp -r dir1 dir2          Copy directory recursively
  cp -p file1 file2        Copy preserving permissions/timestamps
  cp -i file1 file2        Copy interactive (ask before overwrite)
  mv file1 file2           Move / rename file
  mv -i file1 file2        Move interactive (ask before overwrite)
  rm file.txt              Remove file (no trash — GONE FOREVER)
  rm -i file.txt           Remove interactive (ask before each)
  rm -r dir/               Remove directory recursively
  rm -rf dir/              Force remove (no confirmation) — DANGER!
  rmdir empty_dir/         Remove empty directory only
  ln -s target linkname    Create symbolic link (symlink)
  ln target hardlink       Create hard link
  readlink -f symlink      Resolve symlink to real path
  stat file.txt            Show file metadata (size, inode, times)
  file image.bin           Determine file type (magic bytes)
  du -sh dir/              Disk usage of directory (human-readable)
  du -sh *                 Disk usage of each item in current dir
  df -h                    Disk free space on all mounted filesystems
  df -h /home              Free space on the filesystem containing /home
"""
print(commands_files)

# Live demo
print("  Live demo:")
run("touch notes.txt README.md config.yaml")
run("mkdir -p backups/2024")
run("ls -lh")
run("du -sh .")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Viewing and editing text
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Viewing and editing text files")
print("━" * 65)
print()

commands_text = """
  cat file.txt             Print entire file to stdout
  cat -n file.txt          Print with line numbers
  cat file1 file2          Concatenate two files to stdout
  tac file.txt             Print file in reverse (last line first)
  less file.txt            Paginated viewer (q=quit, /=search, n=next)
  more file.txt            Older paginator (less is better)
  head file.txt            First 10 lines
  head -n 20 file.txt      First 20 lines
  head -c 100 file.txt     First 100 bytes
  tail file.txt            Last 10 lines
  tail -n 20 file.txt      Last 20 lines
  tail -f /var/log/syslog  Follow file live (stream new lines as written)
  tail -F file.txt         Follow + reopen if file is rotated
  wc file.txt              Word count: lines words bytes
  wc -l file.txt           Count lines only
  wc -w file.txt           Count words only
  diff file1 file2         Show differences between two files
  diff -u file1 file2      Unified diff format (used for patches)
  diff -r dir1 dir2        Diff two directories recursively
  xxd file.bin             Hex dump a file
"""
print(commands_text)

# Create a sample file for demos
sample = "\n".join([
    "name,age,score",
    "Alice,30,95",
    "Bob,25,87",
    "Charlie,35,92",
    "Diana,28,88",
    "Eve,32,99",
])
with open("data.csv", "w", encoding="utf-8") as f:
    f.write(sample)

print("  Live demo (data.csv):")
run("cat data.csv")
print()
run("wc -l data.csv")
run("head -n 3 data.csv")
run("tail -n 3 data.csv")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: grep — search text
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — grep: pattern searching")
print("━" * 65)
print()

commands_grep = """
  grep "pattern" file      Search for pattern in file
  grep -i "pattern" file   Case-insensitive search
  grep -n "pattern" file   Show line numbers
  grep -v "pattern" file   Invert: lines NOT matching
  grep -r "pattern" dir/   Recursive search in directory
  grep -l "pattern" dir/*  Print filenames with matches (not lines)
  grep -c "pattern" file   Count matching lines
  grep -w "word" file      Match whole word (not substring)
  grep -A 3 "pattern" file Print 3 lines After each match
  grep -B 3 "pattern" file Print 3 lines Before each match
  grep -C 3 "pattern" file Print 3 lines of Context (before+after)
  grep -E "regex" file     Extended regex (alternation, +, ?, {})
  grep -P "regex" file     Perl-compatible regex (lookahead, etc.)
  grep "^word" file        Lines starting with "word" (^ = start of line)
  grep "word$" file        Lines ending with "word" ($ = end of line)
  grep "^$" file           Blank lines
  grep -o "pattern" file   Print only the matching part (not whole line)

  Common regex in grep:
  .     any single character
  *     zero or more of preceding
  +     one or more (with -E or -P)
  ?     zero or one (with -E or -P)
  ^     start of line
  $     end of line
  [abc] character class
  [0-9] digit
  \\d    digit (with -P)
  \\w    word character (with -P)
  \\s    whitespace (with -P)
"""
print(commands_grep)

print("  Live demo:")
run('grep "Alice" data.csv')
run('grep -i "alice" data.csv')
run('grep -n "9[0-9]" data.csv')
run('grep -v "name" data.csv')
run('grep -E "Alice|Eve" data.csv')
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: sed, awk — text transformation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — sed and awk: text transformation")
print("━" * 65)
print()

commands_sed_awk = r"""
  sed COMMANDS:
  sed 's/old/new/' file         Replace first occurrence per line
  sed 's/old/new/g' file        Replace ALL occurrences (global)
  sed 's/old/new/gi' file       Replace all, case-insensitive
  sed -i 's/old/new/g' file     In-place replace (modifies the file)
  sed -i.bak 's/old/new/g' f    In-place + create .bak backup
  sed 's/old/new/2' file        Replace 2nd occurrence per line
  sed '2d' file                 Delete line 2
  sed '2,5d' file               Delete lines 2 through 5
  sed '/pattern/d' file         Delete lines matching pattern
  sed -n '2,5p' file            Print only lines 2–5
  sed -n '/start/,/end/p' file  Print between two patterns
  sed '1s/^/HEADER\n/' file     Insert line before line 1
  sed 's/\s\+/ /g' file         Replace multiple spaces with one
  sed '/^$/d' file              Delete blank lines
  sed 's/^/    /' file          Indent every line by 4 spaces
  sed 's/[0-9]/X/g' file        Replace all digits with X

  awk COMMANDS:
  awk '{print $1}' file         Print first field (space-delimited)
  awk '{print $NF}' file        Print last field (NF = number of fields)
  awk -F, '{print $2}' file     Set comma as field separator
  awk -F: '{print $1}' /etc/passwd   Print usernames (: separated)
  awk 'NR==3' file              Print line 3 (NR = record/line number)
  awk 'NR>=2 && NR<=5' file     Print lines 2 through 5
  awk '/pattern/' file          Print lines matching pattern
  awk '!/pattern/' file         Print lines NOT matching pattern
  awk '{sum+=$3} END{print sum}' f   Sum column 3, print total
  awk '{count[$1]++} END{for(k in count) print k, count[k]}' f  Count by field 1
  awk 'length($0) > 80' file    Print lines longer than 80 chars
  awk '{print NR, $0}' file     Prefix every line with line number
  awk -F, 'NR>1 {print $1,$3}' f.csv   Print cols 1,3 (skip header)
  awk '{$1=$1; print}' file     Remove extra whitespace between fields
"""
print(commands_sed_awk)

print("  Live demo:")
run("sed 's/,/|/g' data.csv")
print()
run("awk -F, 'NR>1 {print $1, \"age:\", $2, \"score:\", $3}' data.csv")
print()
run("awk -F, 'NR>1 {sum+=$3} END {print \"Average score:\", sum/(NR-1)}' data.csv")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: sort, uniq, cut, tr, paste
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — sort, uniq, cut, tr, paste")
print("━" * 65)
print()

commands_transform = """
  sort file                Sort lines alphabetically
  sort -r file             Sort reverse
  sort -n file             Numeric sort (10 before 20, not 1 before 2)
  sort -k2 file            Sort by field 2
  sort -t, -k3 -n file.csv Sort CSV by column 3 numerically
  sort -u file             Sort + remove duplicates (unique)
  sort -h file             Human-readable sort (1K < 1M < 1G)

  uniq file                Remove ADJACENT duplicate lines
  uniq -c file             Count occurrences
  uniq -d file             Print only duplicate lines
  uniq -u file             Print only unique lines
  sort file | uniq -c | sort -rn   Classic: count + rank by frequency

  cut -d, -f2 file.csv     Cut: delimiter comma, field 2
  cut -d: -f1,3 /etc/passwd  Fields 1 and 3 from colon-delimited
  cut -c1-10 file          Characters 1 through 10
  cut -c-5 file            First 5 characters

  tr 'a-z' 'A-Z'           Translate lower to uppercase
  tr -d '\\n'               Delete newlines (join lines)
  tr -s ' '                Squeeze: multiple spaces → one
  tr -dc '0-9\\n'           Delete all non-digits (keep digits and newlines)
  echo "hello" | tr 'el' 'ip'  → hippo (char-by-char replacement)

  paste file1 file2        Merge files side by side (tab-separated)
  paste -d, file1 file2    Merge with comma separator
  paste -s file            Join all lines of a file with tab separator

  column -t file           Align columns for pretty printing
  column -t -s, file.csv   Align CSV columns
"""
print(commands_transform)

print("  Live demo:")
run("sort -t, -k3 -nr data.csv")
print()
run("cut -d, -f1,3 data.csv")
print()
run("awk -F, 'NR>1 {print $2}' data.csv | sort -n")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 7: find — locate files
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 7 — find: locate files by any criterion")
print("━" * 65)
print()

commands_find = """
  find . -name "*.py"          Find all Python files in current dir
  find /home -name "*.log"     Find .log files in /home
  find . -iname "*.PY"         Case-insensitive name match
  find . -type f               Only regular files (not dirs)
  find . -type d               Only directories
  find . -type l               Only symlinks
  find . -size +10M            Files larger than 10 megabytes
  find . -size -100k           Files smaller than 100 kilobytes
  find . -size +1G             Files larger than 1 GB
  find . -mtime -7             Modified in last 7 days
  find . -mtime +30            Modified more than 30 days ago
  find . -newer reference.txt  Modified more recently than reference.txt
  find . -empty                Empty files or directories
  find . -perm 755             Files with exactly permission 755
  find . -perm /u+x            Files where owner has execute bit
  find . -user alice           Files owned by alice
  find . -group admin          Files owned by admin group
  find . -maxdepth 2 -name "*.py"  Search max 2 levels deep
  find . -mindepth 2 -name "*.py"  Skip top level (at least 2 deep)

  find + actions:
  find . -name "*.tmp" -delete          Delete found files
  find . -name "*.log" -exec rm {} \\;   Delete with exec (slower)
  find . -name "*.py" -exec wc -l {} +  Count lines in all .py files
  find . -name "*.txt" | xargs grep "pattern"  Search inside found files
  find . -type f -print0 | xargs -0 chmod 644  Handle filenames with spaces
  find . -name "core" -o -name "*.dump"        OR condition

  xargs:
  find . -name "*.log" | xargs rm -f           Remove all .log files
  echo "file1 file2 file3" | xargs -I{} cp {} /backup/  Placeholder
  find . -name "*.py" | xargs -P4 pylint        Run 4 parallel linters
"""
print(commands_find)

print("  Live demo:")
run("find . -type f -name '*.csv'")
run("find . -type d")
run("find . -newer notes.txt")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 8: Archiving and compression
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 8 — Archiving and compression")
print("━" * 65)
print()

commands_archive = """
  TAR (Tape ARchive — the standard):
  tar -czf archive.tar.gz dir/    Create .tar.gz (c=create, z=gzip, f=file)
  tar -cjf archive.tar.bz2 dir/  Create .tar.bz2 (bzip2 — slower, smaller)
  tar -cJf archive.tar.xz dir/   Create .tar.xz (xz — smallest, slowest)
  tar -cvzf archive.tar.gz dir/  Create + verbose (see filenames)
  tar -tzf archive.tar.gz        List contents of .tar.gz
  tar -xzf archive.tar.gz        Extract .tar.gz (current directory)
  tar -xzf archive.tar.gz -C /tmp  Extract to /tmp
  tar -xzf archive.tar.gz file   Extract one specific file
  tar -czf backup.tar.gz --exclude='*.log' dir/   Exclude .log files

  GZIP (compress single files):
  gzip file.txt                  Compress → file.txt.gz (original deleted)
  gzip -k file.txt               Compress + keep original (-k = keep)
  gzip -d file.txt.gz            Decompress (same as gunzip)
  gunzip file.txt.gz             Decompress
  gzip -l file.txt.gz            List compression ratio
  zcat file.txt.gz               View compressed file without extracting
  zgrep "pattern" file.txt.gz    Grep inside compressed file

  ZIP (cross-platform, common with Windows):
  zip archive.zip file1 file2    Create zip archive
  zip -r archive.zip dir/        Zip a directory recursively
  zip -P password arch.zip file  Password-protected zip
  unzip archive.zip              Extract zip
  unzip archive.zip -d /tmp      Extract to /tmp
  unzip -l archive.zip           List contents without extracting

  OTHER:
  bzip2 file.txt                 Compress with bzip2
  bunzip2 file.txt.bz2           Decompress bzip2
  xz file.txt                    Compress with xz (strong compression)
  unxz file.txt.xz               Decompress xz
  7z a archive.7z dir/           7-Zip archive (need p7zip package)
  7z x archive.7z                Extract 7z archive
"""
print(commands_archive)

print("  Live demo:")
run("tar -czf backup.tar.gz .")
run("ls -lh backup.tar.gz")
run("tar -tzf backup.tar.gz")
print()

# Cleanup
import shutil
os.chdir(tempfile.gettempdir())   # cross-platform: works on Windows + Linux/macOS
shutil.rmtree(DEMO)
print(f"  Cleaned up demo directory.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · System Administration — Processes, Permissions, and Monitoring": {
        "description": (
            "Linux system administration commands. "
            "Process management: ps, top, htop, kill, nice. "
            "User and permission management: chmod, chown, useradd. "
            "System monitoring: free, vmstat, iostat, df, dmesg. "
            "Network commands: ss, netstat, curl, wget, ping, traceroute. "
            "Disk management: lsblk, fdisk, mount, fstab."
        ),
        "language": "python",
        "code": r'''
import subprocess
import os

print("=" * 65)
print("  SYSTEM ADMINISTRATION — PROCESSES, PERMISSIONS, MONITORING")
print("=" * 65)
print()

def run(cmd, show_output=True):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8")
    output = result.stdout.strip() or result.stderr.strip()
    if show_output and output:
        for line in output.split('\n')[:20]:
            print(f"    {line}")
    return output

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Process management
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Process management")
print("━" * 65)
print()

PROCESS_COMMANDS = """
  VIEWING PROCESSES:
  ps aux               All processes (a=all users, u=user format, x=no tty)
  ps aux | grep python Find python processes
  ps -ef               All processes, full format (shows PPID)
  ps --forest          Show parent-child hierarchy as a tree
  ps -p 1234           Info for specific PID
  pstree               Process tree (visual)
  pstree -p            Process tree with PIDs
  pgrep nginx          Find PID(s) of processes named nginx
  pgrep -a python      Find PIDs of python + show full command
  pidof sshd           Get PID of sshd

  top                  Live process viewer (interactive)
                         q=quit, k=kill, r=renice, 1=per-CPU, M=sort by mem
  htop                 Enhanced top (needs htop package — recommended)
  btop                 Modern beautiful process manager
  atop                 Advanced system + process monitor

  top shortcuts inside top:
    P    Sort by CPU usage
    M    Sort by memory usage
    T    Sort by cumulative CPU time
    k    Kill a process (prompts for PID)
    r    Renice a process (change priority)
    1    Toggle per-CPU view
    q    Quit

  KILLING PROCESSES:
  kill 1234            Send SIGTERM (15) to PID 1234 — polite
  kill -9 1234         Send SIGKILL — force (use as LAST RESORT)
  kill -HUP 1234       Send SIGHUP — conventionally: reload config
  kill -STOP 1234      Pause a process (SIGSTOP)
  kill -CONT 1234      Resume a paused process (SIGCONT)
  killall nginx        Kill all processes named nginx (SIGTERM)
  killall -9 nginx     Force kill all nginx processes
  pkill -f "python train.py"   Kill by matching command line

  PROCESS PRIORITY:
  nice -n 10 command   Start command with lower priority (nice 10)
  nice -n -10 command  Higher priority (requires root for negative)
  renice 15 -p 1234    Change priority of running process 1234
  renice -5 -p 1234    Increase priority (requires root)

  BACKGROUND JOBS:
  command &            Run in background
  jobs                 List background jobs
  fg %1                Bring job 1 to foreground
  bg %1                Resume job 1 in background
  Ctrl+Z               Suspend foreground job
  Ctrl+C               Interrupt (SIGINT) foreground job
  disown %1            Detach job from shell (survives shell exit)
  nohup command &      Run immune to hangup (survives logout)
  screen -S mysession  Start a named screen session
  tmux new -s mywork   Start a named tmux session (prefer tmux)
  tmux attach -t mywork  Reattach to existing tmux session
"""
print(PROCESS_COMMANDS)

print("  Live demo:")
run("ps aux | head -5")
print()
run("ps aux --sort=-%cpu | head -6")
print()
run("pgrep -a python | head -5")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: File permissions and ownership
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Permissions and ownership")
print("━" * 65)
print()

PERM_COMMANDS = r"""
  CHMOD — Change file permissions:
  chmod 755 file        rwxr-xr-x  (owner all, group/others r+x)
  chmod 644 file        rw-r--r--  (owner r+w, others read-only)
  chmod 600 file        rw-------  (owner only — private files)
  chmod 700 dir/        rwx------  (directory: owner only)
  chmod 777 file        rwxrwxrwx  (everyone all — AVOID IN PRODUCTION)
  chmod +x script.sh    Add execute for all (relative change)
  chmod -x script.sh    Remove execute for all
  chmod u+x script.sh   Add execute for user/owner only
  chmod g-w file        Remove write from group
  chmod o-r file        Remove read from others
  chmod a+r file        Add read for all (a = all: u+g+o)
  chmod ug=rw file      Set user and group to exactly rw
  chmod -R 755 dir/     Apply recursively to directory tree
  chmod u+s program     Set SUID bit (run as owner)
  chmod g+s dir/        Set SGID on directory (inherit group)
  chmod +t /tmp         Set sticky bit

  Symbolic vs numeric:
    Symbolic: chmod u+x, chmod go-w, chmod a=r
    Numeric:  sum of r=4, w=2, x=1 for each triad
    Example:  rwxr-xr-- = 7(owner) 5(group) 4(others) = 754

  CHOWN — Change owner/group:
  chown alice file             Change owner to alice
  chown alice:devs file        Change owner to alice, group to devs
  chown :devs file             Change group only
  chown -R alice:devs dir/     Recursive ownership change
  chgrp devs file              Change group only (alternative to chown)

  VIEWING PERMISSIONS:
  ls -la file              Show permissions + owner + group
  stat file                Detailed file metadata including oct perms
  getfacl file             Show ACL (extended permissions, if enabled)
  setfacl -m u:bob:rx file Set ACL: give bob read+execute on this file

  UMASK — default permission mask:
  umask                    Show current mask (e.g., 0022)
  umask 022                Files: 644 (666-022), Dirs: 755 (777-022)
  umask 027                Files: 640, Dirs: 750 (more restrictive)
  umask 077                Files: 600, Dirs: 700 (private)
  # umask is the bits REMOVED from the default (666 for files, 777 for dirs)
"""
print(PERM_COMMANDS)

print("  Live demo:")
import tempfile
tf = tempfile.NamedTemporaryFile(delete=False, suffix=".sh")
tf.write(b"#!/bin/bash\necho hello\n")
tf.close()
run(f"ls -la {tf.name}")
os.chmod(tf.name, 0o755)
run(f"ls -la {tf.name}")
os.unlink(tf.name)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: System monitoring
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — System monitoring and resource usage")
print("━" * 65)
print()

MONITOR_COMMANDS = """
  MEMORY:
  free -h              Memory usage (human-readable: MB/GB)
  free -m              Memory usage in megabytes
  cat /proc/meminfo    Detailed memory information
  vmstat 1             Virtual memory stats every 1 second
  vmstat -s            Memory stats summary

  CPU:
  uptime               Load averages (1, 5, 15 minute)
  w                    Who is logged in + system load
  mpstat               Per-CPU statistics (from sysstat package)
  mpstat -P ALL 1      All CPU cores, every 1 second
  cat /proc/cpuinfo    Detailed CPU information
  nproc                Number of CPU cores available
  lscpu                CPU architecture and features

  DISK I/O:
  iostat               I/O statistics (from sysstat)
  iostat -x 1          Extended disk stats every 1 second
  iotop                Live per-process I/O monitor (needs root)
  lsblk                List block devices (disks, partitions)
  lsblk -f             Show filesystems + UUIDs
  blkid                Show device UUIDs and filesystem types
  df -h                Filesystem disk usage
  df -i                Filesystem inode usage
  du -sh /var/log/     Directory size
  du -sh * | sort -h   Sizes of all items, sorted

  SYSTEM INFO:
  uname -a             Kernel version, hostname, architecture
  uname -r             Just the kernel version
  lsb_release -a       Distro name and version (Debian/Ubuntu)
  cat /etc/os-release  Distro info (works on all distros)
  hostname             Show hostname
  hostname -I          Show all IP addresses
  date                 Current date and time
  timedatectl          Timezone + NTP sync status
  uptime               How long system has been running
  last                 Login history
  who                  Who is currently logged in
  lspci                List PCI devices (GPU, NICs, etc.)
  lsusb                List USB devices
  lshw                 List hardware (needs lshw package)
  dmidecode            BIOS and hardware info (needs root)
  sensors              CPU/GPU temperature (lm-sensors package)
  dmesg                Kernel ring buffer (hardware events, errors)
  dmesg -H             Kernel messages, human-readable timestamps
  dmesg --level=err    Show only errors
  dmesg | grep -i gpu  Filter for GPU-related messages
"""
print(MONITOR_COMMANDS)

print("  Live demo:")
run("uname -a")
run("uptime")
run("free -h")
run("df -h / | tail -1")
run("nproc")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Network commands
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Networking commands")
print("━" * 65)
print()

NET_COMMANDS = r"""
  CONNECTIVITY:
  ping host            Send ICMP packets (Ctrl+C to stop)
  ping -c 4 host       Send exactly 4 packets
  ping -i 0.2 host     Ping every 0.2 seconds (fast)
  traceroute host      Trace packet path (each hop)
  tracepath host       Like traceroute, no root needed
  mtr host             Live traceroute + ping combined
  nslookup host        DNS lookup (old)
  dig host             DNS lookup (modern, detailed)
  dig +short host      DNS lookup (just the IP)
  dig -x 8.8.8.8       Reverse DNS (IP → hostname)
  host domain.com      Simple DNS lookup
  whois domain.com     WHOIS record for domain

  INTERFACES AND ROUTING:
  ip addr show         All interfaces and IPs (modern)
  ip addr show eth0    Specific interface
  ip link show         Interface link status
  ip route show        Routing table
  ip route get 8.8.8.8 Which interface would reach 8.8.8.8?
  ifconfig             (older — may need net-tools package)
  route -n             (older routing table viewer)
  arp -n               ARP cache (IP → MAC mappings)
  ip neigh show        ARP cache (modern)

  PORTS AND CONNECTIONS:
  ss -tuln             Listening TCP/UDP ports (modern netstat)
  ss -tulnp            Same + show which process
  ss -s                Socket summary statistics
  ss -ta               All TCP connections
  netstat -tuln        (older — may need net-tools)
  netstat -tulnp       (older + process names)
  lsof -i :8080        What process is using port 8080?
  lsof -i tcp          All TCP connections with process names
  fuser 8080/tcp       PID using port 8080

  HTTP TOOLS:
  curl https://url     Fetch URL to stdout
  curl -o file.zip URL Download to file
  curl -L URL          Follow redirects
  curl -I URL          Show headers only (HEAD request)
  curl -H "Auth: token" URL  Set request header
  curl -X POST -d '{"key":"val"}' -H "Content-Type: application/json" URL
  curl -u user:pass URL      Basic auth
  curl -s URL | jq .         Fetch JSON and pretty-print with jq
  curl -v URL                Verbose (show all headers + SSL handshake)

  wget https://url     Download file (continues interrupted downloads)
  wget -c URL          Resume interrupted download
  wget -q URL          Quiet mode
  wget -r -np URL      Recursive download (no parent directories)
  wget --mirror URL    Mirror a website

  TRANSFER:
  scp file user@host:/path        Copy file to remote
  scp user@host:/path/file .      Copy file from remote
  scp -r dir/ user@host:/path/    Copy directory to remote
  rsync -avz dir/ user@host:path/ Sync directory to remote
  rsync -avz --delete dir/ dest/  Sync (delete files not in source)
  rsync -avz --progress src/ dst/ Show progress during sync
  sftp user@host                  Interactive secure FTP session
"""
print(NET_COMMANDS)

print("  Live demo:")
run("ip addr show lo")
print()
run("ss -tuln | head -10")
print()
run("curl -s --max-time 3 https://httpbin.org/ip 2>/dev/null || echo '  (no internet or timeout)'")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: User management
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — User and group management")
print("━" * 65)
print()

USER_COMMANDS = """
  USERS:
  whoami               Current username
  id                   UID, GID, and all groups of current user
  id alice             UID, GID, and groups for alice
  users                List logged-in users (space-separated)
  who                  Who is logged in (terminal, login time, IP)
  w                    Who + what they're doing + system load
  last                 Login history (from /var/log/wtmp)
  lastlog              Last login for all users
  finger alice         User info (if finger is installed)

  ADDING/MODIFYING USERS:
  sudo useradd alice                   Create user (no home, no password)
  sudo useradd -m alice                Create user with home directory
  sudo useradd -m -s /bin/bash alice   Set default shell
  sudo useradd -m -G sudo,docker alice Add to groups
  sudo passwd alice                    Set/change alice's password
  sudo usermod -aG docker alice        Add alice to docker group (append)
  sudo usermod -s /bin/zsh alice       Change default shell
  sudo usermod -l newname oldname      Rename user
  sudo usermod -d /newhome -m alice    Move home directory
  sudo userdel alice                   Delete user (keep home)
  sudo userdel -r alice                Delete user + home directory
  sudo chsh -s /bin/zsh alice          Change shell (user can run for self)
  chsh -s /bin/zsh                     Change own shell

  GROUPS:
  groups                     Show current user's groups
  groups alice               Show alice's groups
  sudo groupadd developers   Create a new group
  sudo groupdel developers   Delete a group
  sudo gpasswd -a alice devs Add alice to devs group
  sudo gpasswd -d alice devs Remove alice from devs group
  getent group docker        List members of docker group
  cat /etc/group | grep alice  Find all groups alice is in

  SWITCHING USERS:
  su alice             Switch to alice (needs alice's password)
  su - alice           Switch + load alice's full environment
  su -                 Switch to root (needs root password)
  sudo -i              Open root shell (uses YOUR password + sudo)
  sudo -u alice cmd    Run cmd as alice
  sudo su - alice      Become alice via sudo (root required)
"""
print(USER_COMMANDS)

print("  Live demo:")
run("whoami")
run("id")
run("groups")
print()
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Shell Scripting, Shortcuts & Power User Commands": {
        "description": (
            "Shell scripting fundamentals and power user techniques. "
            "Bash scripting: variables, conditionals, loops, functions. "
            "Keyboard shortcuts for terminal efficiency. "
            "History, aliases, and shell customisation. "
            "Advanced commands: xargs, parallel, jq, screen/tmux. "
            "Cron jobs for scheduling. vim basics."
        ),
        "language": "python",
        "code": r'''
import subprocess
import os
import tempfile
import stat

print("=" * 65)
print("  SHELL SCRIPTING, SHORTCUTS & POWER USER COMMANDS")
print("=" * 65)
print()

def run(cmd, show_output=True):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                encoding="utf-8", timeout=5)
        output = result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        print(f"    [skipped — command timed out: {cmd[:60]}]")
        return ""
    except Exception as e:
        print(f"    [skipped — {e}]")
        return ""
    if show_output and output:
        for line in output.split('\n')[:20]:
            print(f"    {line}")
    return output

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Bash scripting
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Bash scripting fundamentals")
print("━" * 65)
print()

SCRIPT_TEMPLATE = r"""#!/bin/bash


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BASH SCRIPT TEMPLATE — annotated reference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# SHEBANG: tells the kernel which interpreter to use
# /bin/bash for bash-specific features
# /bin/sh for maximum portability (POSIX only)

# EXIT ON ERROR — critical for production scripts
set -e           # exit immediately if any command fails
set -u           # exit on unset variable (catches typos)
set -o pipefail  # catch errors in pipes (cmd1 | cmd2)
set -x           # print every command before executing (debug mode)
# Short form: set -euo pipefail

# ── Variables ───────────────────────────────────────────────────────────
NAME="Alice"
AGE=30
TODAY=$(date +%Y-%m-%d)    # command substitution
PI=3.14159

echo "Name: $NAME, Age: $AGE, Today: $TODAY"
echo "String length: ${#NAME}"       # length of $NAME
echo "Upper: ${NAME^^}"              # uppercase (bash 4+)
echo "Default: ${UNSET_VAR:-hello}"  # use default if unset
echo "Error if unset: ${NAME:?Variable required}"

# ── Conditionals ─────────────────────────────────────────────────────────
if [ "$AGE" -ge 18 ]; then
    echo "Adult"
elif [ "$AGE" -ge 13 ]; then
    echo "Teenager"
else
    echo "Child"
fi

# [[ ]] (bash-specific, more powerful than [ ]):
if [[ "$NAME" == "Alice" && "$AGE" -gt 20 ]]; then
    echo "Alice is over 20"
fi

# File tests:
# -f file     regular file exists
# -d dir      directory exists
# -e path     path exists (any type)
# -r file     file is readable
# -w file     file is writable
# -x file     file is executable
# -s file     file is non-empty
# -z "$var"   string is empty
# -n "$var"   string is non-empty

if [[ -f "/etc/passwd" ]]; then
    echo "/etc/passwd exists"
fi

# ── Arithmetic ───────────────────────────────────────────────────────────
RESULT=$(( 10 + 5 * 3 ))   # bash arithmetic
echo "Result: $RESULT"
((COUNT++))                  # increment in place
echo "Count: $COUNT"

# For floating point, use bc or awk:
echo "scale=2; 22/7" | bc
awk 'BEGIN { printf "Pi approx: %.5f\n", 22/7 }'

# ── Loops ────────────────────────────────────────────────────────────────
# C-style for loop
for ((i=1; i<=5; i++)); do
    echo "Iteration $i"
done

# Iterate over list
for name in Alice Bob Charlie; do
    echo "Hello, $name!"
done

# Iterate over files
for file in *.py; do
    echo "Processing: $file"
done

# Iterate over array
FRUITS=("apple" "banana" "cherry")
for fruit in "${FRUITS[@]}"; do
    echo "Fruit: $fruit"
done

# Range with step
for i in {1..10..2}; do   # 1 3 5 7 9
    echo -n "$i "
done; echo

# While loop
COUNT=0
while [[ $COUNT -lt 5 ]]; do
    echo "Count: $COUNT"
    ((COUNT++))
done

# Read file line by line
while IFS= read -r line; do
    echo "Line: $line"
done < /etc/hostname

# Until loop (opposite of while)
until [[ -f /tmp/done.flag ]]; do
    sleep 1
done

# ── Functions ────────────────────────────────────────────────────────────
greet() {
    local name="$1"   # local variable (doesn't pollute global scope)
    local greeting="${2:-Hello}"   # default value
    echo "$greeting, $name!"
}
greet "Alice"
greet "Bob" "Hi"

# Function with return value
add() { echo $(( $1 + $2 )); }
result=$(add 3 4)
echo "3 + 4 = $result"

# Return exit codes
is_even() {
    (( $1 % 2 == 0 ))   # exit code 0 (success) if true
}
if is_even 4; then echo "4 is even"; fi

# ── Arrays ───────────────────────────────────────────────────────────────
ARR=(one two three)
echo "${ARR[0]}"       # first element: one
echo "${ARR[-1]}"      # last element: three
echo "${ARR[@]}"       # all elements
echo "${#ARR[@]}"      # number of elements
ARR+=(four)            # append element
echo "${ARR[@]:1:2}"   # slice: elements 1 and 2

# Associative array (bash 4+, dict-like)
declare -A COLORS
COLORS[red]="#FF0000"
COLORS[green]="#00FF00"
echo "${COLORS[red]}"
echo "${!COLORS[@]}"   # all keys

# ── Error handling ───────────────────────────────────────────────────────
cleanup() {
    echo "Cleaning up..."
    rm -f /tmp/mytemp.*
}
trap cleanup EXIT      # run cleanup() on ANY exit (success or error)
trap 'echo "Error at line $LINENO"' ERR  # run on error

command_that_might_fail || {
    echo "Command failed, handling..."
    exit 1
}

# Check exit code explicitly:
if ! grep "pattern" file.txt; then
    echo "Pattern not found"
fi

# ── String manipulation ──────────────────────────────────────────────────
STR="Hello, World!"
echo "${STR:7}"         # substring from pos 7: World!
echo "${STR:7:5}"       # 5 chars from pos 7: World
echo "${STR/World/Linux}"  # replace first occurrence
echo "${STR//l/L}"         # replace all occurrences
echo "${STR%!}"            # strip trailing ! (shortest match)
echo "${STR%%,*}"          # strip from first comma to end
echo "${STR#Hello, }"      # strip prefix
echo "${#STR}"             # length: 13
STR_UPPER="${STR^^}"       # HELLO, WORLD!
STR_LOWER="${STR,,}"       # hello, world!

# ── Script arguments ─────────────────────────────────────────────────────
# $0 = script name
# $1 $2 ... = arguments
# $@ = all arguments (as separate words — use this!)
# $* = all arguments (as one string — rarely correct)
# $# = number of arguments
# shift   = shift $1 away, $2 becomes $1, etc.

# Argument parsing pattern:
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)    echo "Usage: $0 [-v] [-o output]"; exit 0 ;;
        -v|--verbose) VERBOSE=true; shift ;;
        -o|--output)  OUTPUT="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done
"""

print("  Complete bash script template (annotated):")
print()
for line in SCRIPT_TEMPLATE.split('\n')[:80]:  # show first 80 lines
    print(f"  {line}")
print(f"  ... (full script has {len(SCRIPT_TEMPLATE.split(chr(10)))} lines)")
print()

# Write and run a simple demo script
demo_script = r"""#!/bin/bash
set -euo pipefail

# Count files by extension in current directory
echo "File extension summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━"

declare -A counts
for file in /etc/*; do
    if [[ -f "$file" ]]; then
        ext="${file##*.}"
        if [[ "$ext" == "$file" ]]; then
            ext="(no extension)"
        fi
        ((counts[$ext]++)) || counts[$ext]=1
    fi
done

for ext in "${!counts[@]}"; do
    printf "  %-20s %d files\n" "$ext" "${counts[$ext]}"
done | sort -k2 -rn | head -10
"""

with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, encoding="utf-8") as f:
    f.write(demo_script)
    script_path = f.name

import platform
os.chmod(script_path, stat.S_IRWXU)
if platform.system() == "Windows":
    print("  [bash demo skipped — bash not natively available on Windows]")
    print("  Tip: install Git Bash or enable WSL to run bash scripts.")
else:
    print("  Running demo script (count /etc file extensions):")
    try:
        result = subprocess.run(["bash", script_path], capture_output=True,
                                text=True, encoding="utf-8", timeout=10)
        print(result.stdout[:600] if result.stdout else result.stderr[:200])
    except subprocess.TimeoutExpired:
        print("  [bash demo timed out after 10s]")
    except FileNotFoundError:
        print("  [bash not found — install Git Bash or WSL to run bash scripts]")
try:
    os.unlink(script_path)
except OSError:
    pass
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Keyboard shortcuts
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Terminal keyboard shortcuts (must memorise)")
print("━" * 65)
print()

SHORTCUTS = """
  CURSOR MOVEMENT (readline / bash):
  Ctrl+A        Move to beginning of line
  Ctrl+E        Move to end of line
  Ctrl+F        Move forward one character
  Ctrl+B        Move backward one character
  Alt+F         Move forward one word
  Alt+B         Move backward one word

  EDITING:
  Ctrl+D        Delete character under cursor (also: exit if line empty)
  Ctrl+H        Delete character before cursor (= Backspace)
  Ctrl+K        Kill (cut) from cursor to end of line
  Ctrl+U        Kill from cursor to beginning of line
  Ctrl+W        Kill word before cursor
  Alt+D         Kill word after cursor
  Ctrl+Y        Yank (paste) killed text
  Ctrl+_        Undo last edit
  Alt+T         Swap current word with previous word

  HISTORY:
  Ctrl+P        Previous command (= Up arrow)
  Ctrl+N        Next command (= Down arrow)
  Ctrl+R        Reverse history search (type to search, Ctrl+R again = older)
  Ctrl+G        Abort history search, keep current command
  !!            Repeat last command
  !n            Run command number n from history
  !-n           Run command n from end of history
  !string       Run most recent command starting with "string"
  !$            Last argument of previous command
  !*            All arguments of previous command
  ^old^new      Run last command with "old" replaced by "new"

  CONTROL:
  Ctrl+C        Send SIGINT — interrupt/kill foreground process
  Ctrl+Z        Send SIGTSTP — suspend foreground process
  Ctrl+D        EOF — exit shell / finish input
  Ctrl+L        Clear screen (same as: clear)
  Ctrl+S        Stop output (XOFF — terminal stops showing output)
  Ctrl+Q        Resume output (XON)

  COMPLETION (Tab):
  Tab           Complete command/file/variable
  Tab Tab       Show all completions (when ambiguous)
  Alt+?         Show possible completions
  Alt+*         Insert all completions

  HISTORY TIPS:
  history          Show numbered command history
  history | grep apt   Find all apt commands you've run
  history -c       Clear history
  export HISTSIZE=10000          Keep 10000 commands
  export HISTFILESIZE=20000      Keep 20000 in history file
  export HISTCONTROL=ignoredups  Don't record duplicate commands
  export HISTIGNORE="ls:cd:pwd"  Don't record these commands
  export HISTTIMEFORMAT="%Y-%m-%d %T "  Record timestamps
"""
print(SHORTCUTS)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Power user commands
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Power user commands")
print("━" * 65)
print()

POWER_COMMANDS = r"""
  XARGS — Build and execute commands from standard input:
  find . -name "*.log" | xargs rm          Delete all found .log files
  cat urls.txt | xargs -I{} curl -O {}     Download each URL
  find . -name "*.py" | xargs wc -l        Count lines in all .py files
  echo "file1 file2 file3" | xargs -n1 cp -t /dest/   Copy one at a time
  find . -name "*.jpg" | xargs -P4 convert {} {}.png   4 parallel converts
  find . -print0 | xargs -0 grep "pattern" Handle filenames with spaces

  JQ — JSON processor (like awk for JSON):
  curl -s api.example.com | jq .           Pretty-print JSON
  curl -s api.example.com | jq '.items[]'  Iterate array
  cat data.json | jq '.users[] | .name'    Extract field
  cat data.json | jq 'select(.age > 30)'   Filter
  cat data.json | jq -r '.name'            Raw output (no quotes)
  cat data.json | jq '{name:.name, age:.age}'  Transform object
  cat data.json | jq '[.[] | {name, score}]'   Map array

  PARALLEL — GNU parallel (parallel execution):
  parallel echo ::: a b c d              Run echo for each (like xargs)
  parallel -j4 gzip ::: *.txt           Compress 4 files at a time
  parallel wget ::: url1 url2 url3       Download URLs in parallel
  cat jobs.txt | parallel -j8 ./process  8-way parallel job queue
  seq 100 | parallel -j$(nproc) my_cmd  One job per CPU core

  AWK ONE-LINERS (more examples):
  awk '{print NR": "$0}' file              Number lines
  awk 'NR%2==0' file                      Print even lines
  awk '!seen[$0]++' file                  Remove duplicates (preserving order)
  awk '{print $NF}' file                  Print last field
  awk 'length>80' file                    Lines longer than 80 chars
  awk -v n=5 'NR==n' file                 Print line n
  awk 'END{print NR}' file                Count lines (like wc -l)
  awk '{a[$1]+=$2} END{for(k in a) print k,a[k]}' f  Group sum

  SED ONE-LINERS (more examples):
  sed -n '10,20p' file                    Print lines 10-20
  sed 'G' file                            Double-space a file
  sed '/^#/d' file                        Remove comment lines
  sed 's/[[:space:]]*$//' file            Remove trailing whitespace
  sed ':a;N;$!ba;s/\n/ /g' file           Join all lines into one
  sed '1!G;h;$!d' file                    Reverse line order (like tac)

  STRING TOOLS:
  echo "hello world" | tr ' ' '\n' | sort | uniq -c | sort -rn
                                          Word frequency count
  base64 file.txt                         Base64 encode
  base64 -d encoded.txt                   Base64 decode
  md5sum file.txt                         MD5 checksum
  sha256sum file.txt                      SHA-256 checksum
  sha256sum -c checksums.txt              Verify checksums from file

  PROCESS SUBSTITUTION AND PIPES:
  diff <(ls dir1/) <(ls dir2/)            Compare directory listings
  comm <(sort file1) <(sort file2)        Lines common to / unique to each
  wc -l < <(find . -name "*.py")         Count Python files
  tee file.txt | wc -l                   Write to file AND stdout
  command | tee -a log.txt               Append to log and display
"""
print(POWER_COMMANDS)

print("  Live demo:")
run("echo '1 apple\n2 banana\n3 cherry' | awk '{print $2, \"(id=$1)\"}'")
print()
run("seq 1 5 | xargs -I{} echo 'Item {}'")
print()
run("echo 'the quick brown fox' | tr ' ' '\n' | sort | uniq -c | sort -rn")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Cron jobs and scheduling
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Cron jobs: scheduling tasks")
print("━" * 65)
print()

CRON_GUIDE = r"""
  CRON SYNTAX:
  ┌─────────── minute    (0-59)
  │ ┌───────── hour      (0-23)
  │ │ ┌─────── day       (1-31)
  │ │ │ ┌───── month     (1-12)
  │ │ │ │ ┌─── weekday   (0-7, 0 and 7 = Sunday)
  │ │ │ │ │
  * * * * *  command

  Special values:
  *     any value
  ,     list: 1,3,5
  -     range: 1-5
  /     step: */5 = every 5

  EXAMPLES:
  */5 * * * *      command    Every 5 minutes
  0 * * * *        command    Every hour (at :00)
  0 0 * * *        command    Daily at midnight
  0 2 * * *        command    Daily at 2 AM
  0 2 * * 0        command    Weekly on Sunday at 2 AM
  0 0 1 * *        command    Monthly on 1st at midnight
  0 8 * * 1-5      command    Weekdays at 8 AM
  0 0 1 1 *        command    Yearly on Jan 1st at midnight
  30 6 1,15 * *    command    At 6:30 on 1st and 15th of month

  MANAGING CRONTABS:
  crontab -e        Edit your crontab (use EDITOR env var to set editor)
  crontab -l        List your crontab
  crontab -r        Remove your crontab (careful — no confirmation!)
  sudo crontab -u alice -e  Edit alice's crontab

  SYSTEM CRON:
  /etc/crontab           System-wide crontab (has username field)
  /etc/cron.d/           Drop-in cron files
  /etc/cron.daily/       Scripts run daily
  /etc/cron.hourly/      Scripts run hourly
  /etc/cron.weekly/      Scripts run weekly
  /etc/cron.monthly/     Scripts run monthly

  CRON BEST PRACTICES:
  1. Always use FULL PATHS (cron has minimal $PATH):
     0 2 * * * /usr/bin/python3 /home/alice/backup.py

  2. Redirect output to a log file:
     0 2 * * * /home/alice/backup.sh >> /home/alice/backup.log 2>&1

  3. Discard output if you don't need it:
     */5 * * * * /home/alice/healthcheck.sh > /dev/null 2>&1

  4. Set MAILTO to receive errors by email (or empty to silence):
     MAILTO=""   (no email)
     MAILTO="alice@example.com"   (email on failure)

  5. Use systemd timers instead of cron for complex scheduling:
     systemctl list-timers       (see all active timers)

  ANACRON (for systems not always on — laptops):
  /etc/anacrontab: like cron but uses delay instead of exact time
  period  delay  job-id  command
  1       5      daily-backup  /home/alice/backup.sh
"""
print(CRON_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: vim essentials
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — vim: essential commands")
print("━" * 65)
print()

VIM_GUIDE = """
  MODES:
  Normal mode:      default — movement and commands
  Insert mode:      typing text (Enter: i, a, o)
  Visual mode:      selecting text (Enter: v, V, Ctrl+V)
  Command mode:     :commands (Enter: :)

  ENTERING INSERT MODE:
  i     Insert before cursor
  I     Insert at beginning of line
  a     Append after cursor
  A     Append at end of line
  o     Open new line below and insert
  O     Open new line above and insert
  Esc   Return to Normal mode

  MOVEMENT (Normal mode):
  h j k l   ← ↓ ↑ →  (arrow keys also work)
  w         Next word (beginning)
  b         Previous word (beginning)
  e         Next word (end)
  0         Beginning of line
  ^         First non-blank character of line
  $         End of line
  gg        Go to first line of file
  G         Go to last line of file
  :n        Go to line n  (e.g., :42)
  Ctrl+F    Page forward (scroll down)
  Ctrl+B    Page backward (scroll up)
  %         Jump to matching bracket/paren/brace

  EDITING (Normal mode):
  x         Delete character under cursor
  dw        Delete word
  dd        Delete entire line
  D         Delete from cursor to end of line
  yy        Yank (copy) line
  yw        Yank word
  p         Paste after cursor
  P         Paste before cursor
  u         Undo
  Ctrl+R    Redo
  .         Repeat last command

  SEARCH AND REPLACE:
  /pattern      Search forward
  ?pattern      Search backward
  n             Next match
  N             Previous match
  :%s/old/new/g     Replace all occurrences in file
  :%s/old/new/gc    Replace with confirmation
  :5,10s/old/new/g  Replace in lines 5-10
  :s/old/new/       Replace first on current line

  FILE OPERATIONS:
  :w        Save (write)
  :w file   Save to new file
  :q        Quit (fails if unsaved changes)
  :q!       Force quit (discard changes)
  :wq       Save and quit
  :x        Save and quit (like :wq but skips if unchanged)
  ZZ        Save and quit (Normal mode shortcut)
  ZQ        Quit without saving (Normal mode shortcut)

  VISUAL MODE:
  v         Character-wise selection
  V         Line-wise selection
  Ctrl+V    Block (column) selection
  After selecting:
  d         Delete selection
  y         Yank (copy) selection
  >         Indent selection
  <         Unindent selection
  :         Apply command to selection

  USEFUL COMMANDS:
  :set number       Show line numbers
  :set nu!          Toggle line numbers
  :set paste        Disable auto-indent (for pasting)
  :set nopaste      Re-enable auto-indent
  :set ignorecase   Case-insensitive search
  gg=G              Auto-indent entire file
  :split file       Horizontal split
  :vsplit file      Vertical split
  Ctrl+W W          Switch between splits
"""
print(VIM_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Useful one-liners and tricks
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Useful one-liners and tricks")
print("━" * 65)
print()

ONELINERS = r"""
  SYSTEM INFORMATION:
  cat /proc/cpuinfo | grep "model name" | uniq   CPU model
  free -h | awk '/^Mem:/ {print $3"/"$2}'         Used/Total RAM
  df -h | awk 'NR>1 {print $5, $6}' | sort -rh | head   Full disks
  ls -lSh /var/log/ | head                        Largest log files
  ps aux | awk '{print $6/1024 " MB\t" $11}' | sort -rn | head  RSS by process

  FIND AND PROCESS:
  find . -name "*.py" | xargs grep -l "import torch"  Find PyTorch scripts
  find /var/log -name "*.log" -mtime +7 -delete   Delete logs >7 days old
  find . -type f -exec chmod 644 {} +             Set all files to 644
  find . -type d -exec chmod 755 {} +             Set all dirs to 755

  NETWORKING TRICKS:
  ss -s                                             Socket statistics summary
  curl -s ipinfo.io/ip                              Your public IP
  curl -s "https://dns.google/resolve?name=google.com&type=A" | jq .
  nc -zv host 443                                   Test if port 443 is open
  nc -l 8080                                        Listen on port 8080 (netcat)
  python3 -m http.server 8080                       Quick HTTP file server

  TEXT TRICKS:
  seq 1 100 | paste -sd+ | bc                  Sum 1 to 100
  date +%s                                     Unix timestamp
  date -d @1700000000                          Convert timestamp to date
  openssl rand -base64 32                      Generate random 32-byte password
  echo "password" | sha256sum                  Hash a string

  PRODUCTIVITY:
  alias ll='ls -alFh'                          Better ls
  alias ..='cd ..'                             Quick up
  alias grep='grep --color=auto'              Colorised grep
  cd () { builtin cd "$@" && ls; }            ls after every cd
  mkcd () { mkdir -p "$1" && cd "$1"; }      mkdir + cd combo
  HISTSIZE=10000; HISTFILESIZE=20000           Bigger history
  export EDITOR=vim                            Set default editor

  DISK AND MEMORY:
  du -sh */ | sort -rh | head -20             Top 20 largest directories
  lsof +D /var/log/ | wc -l                  Files open in /var/log
  ls -lt /proc/ | grep -E "^d" | head        Recently created processes
  cat /proc/loadavg                           System load (raw)

  DEBUGGING:
  strace command                              Trace system calls
  ltrace command                              Trace library calls
  ldd /usr/bin/python3                        Shared libraries of python3
  nm /usr/lib/x86_64-linux-gnu/libc.so.6 | grep malloc   Symbols in libc
"""
print(ONELINERS)

print("  Live demo:")
run("seq 1 10 | paste -sd+ | bc")
run("date +%s")
run("openssl rand -base64 16 2>/dev/null || head -c 16 /dev/urandom | base64")
print()

print("  COMPLETE COMMAND QUICK REFERENCE:")
QUICK_REF = """
  ┌───────────────────────────────────────────────────────────────────────┐
  │ NAVIGATION   │ PROCESSES    │ TEXT         │ NETWORK      │ SYSTEM    │
  ├───────────────────────────────────────────────────────────────────────┤
  │ pwd  ls  cd  │ ps  top  htop│ cat  less    │ ping  curl   │ df  du    │
  │ tree  pushd  │ kill  killall│ grep  sed    │ wget  scp    │ free      │
  │ find  locate │ nice  nohup  │ awk  sort    │ ssh  rsync   │ uname     │
  │ which  type  │ jobs  fg  bg │ cut  tr  wc  │ ip  ss  nmap │ lscpu     │
  │ readlink stat│ pgrep  pkill │ head  tail   │ dig  nslookup│ lsblk     │
  ├───────────────────────────────────────────────────────────────────────┤
  │ FILES        │ PERMISSIONS  │ USERS        │ PACKAGES     │ LOGS      │
  ├───────────────────────────────────────────────────────────────────────┤
  │ cp  mv  rm   │ chmod  chown │ useradd  id  │ apt  dnf     │ journalctl│
  │ ln  touch    │ chgrp  umask │ passwd  su   │ yum  zypper  │ tail -f   │
  │ tar  gzip    │ sudo  visudo │ groups  who  │ pip  snap    │ dmesg     │
  │ zip  unzip   │ getfacl  acl │ w  last      │ rpm  dpkg    │ syslog    │
  │ dd  rsync    │ setfacl      │ whoami  su   │ flatpak      │ auth.log  │
  └───────────────────────────────────────────────────────────────────────┘
"""
print(QUICK_REF)
''',
    },

}

# Dedent all operation code strings — they're indented inside the dict literal,
# so each line has ~20 leading spaces. textwrap.dedent removes the common indent,
# producing clean left-aligned code that runs without IndentationError.
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# RENDER OPERATIONS (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
def render_operations(st, scripts_dir=None, main_script=None):
    """Render all operations with code display and optional run buttons."""
    import streamlit as st  # local import so module stays importable without st

    st.markdown("---")
    st.subheader("⚙️ Operations")

    if scripts_dir is None:
        scripts_dir = None
    if main_script is None:
        main_script = None

    scripts_available = main_script.exists()

    if "tok_step_status" not in st.session_state:
        st.session_state.tok_step_status = {}
    if "tok_step_outputs" not in st.session_state:
        st.session_state.tok_step_outputs = {}

    for op_name, op_data in OPERATIONS.items():
        with st.expander(f"▶️ {op_name}", expanded=False):
            st.markdown(f"**{op_data['description']}**")
            st.markdown("---")
            st.code(op_data["code"], language=op_data.get("language", "python"))


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────
# render_operations() has been removed.  app.py owns all Streamlit rendering
# via its own render_operation() helper and strips callables from topic dicts
# inside load_topics_for() anyway — so a local render function is never called.

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html = ""
    visual_height = 400
    # try:
    #     from supervised.visuals.regression_visual import (   # ← match your exact folder casing
    #         REG_VISUAL_HTML,
    #         REG_VISUAL_HEIGHT,
    #     )
    #     visual_html   = REG_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = REG_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[01_linear_regression.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": visual_html,
        "visual_height": visual_height,
        "complexity": None,  # COMPLEXITY,
        "operations": OPERATIONS,
    }

