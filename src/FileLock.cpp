#include "RadFiled3D/helpers/FileLock.hpp"
#if defined _WIN32 || defined _WIN64
    #include <windows.h>
    #include <filesystem>
    namespace fs = std::filesystem;
#else
    #include <fcntl.h>
    #include <unistd.h>
    #include <filesystem>
    namespace fs = std::filesystem;
#endif
#include <iostream>


using namespace RadFiled3D;


FileLock::FileLock(const std::string& filename, bool should_lock)
{
    if (!should_lock)
        return;
    this->lock_filename = filename + ".lock";
#if defined _WIN32 || defined _WIN64
    hFile = CreateFile(this->lock_filename.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        throw FileLockException("Unable to open the file");
    }

    OVERLAPPED overlapped = { 0 };
    if (!LockFileEx(hFile, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &overlapped)) {
        CloseHandle(hFile);
        throw FileLockException("Unable to lock the file");
    }
#else
    fd = open(this->lock_filename.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        throw FileLockException("Unable to open the file");
    }

    struct flock fl;
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;
    fl.l_pid = getpid();

    if (fcntl(fd, F_SETLKW, &fl) == -1) {
        close(fd);
        throw FileLockException("Unable to lock the file");
    }
#endif
}

FileLock::~FileLock()
{
#if defined _WIN32 || defined _WIN64
    if (hFile != (void*)-1) {
        CloseHandle(hFile);
    }
#else
    if (fd != -1) {
        close(fd);
    }
#endif
    try {
        if (fs::exists(this->lock_filename)) {
            fs::remove(this->lock_filename);
        }
    }
    catch (const fs::filesystem_error& e) {
			// NOP
    }
}