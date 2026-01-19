#pragma once
#include <string>
#include <stdexcept>
#include "RadFiled3D/helpers/Typing.hpp"


namespace RadFiled3D {
    class RADFILED_API FileLockException : public std::runtime_error {
    public:
        FileLockException(const std::string& message) : std::runtime_error("FileLockException: " + message) {}
    };


    class FileLock {
    public:
        FileLock(const std::string& filename, bool should_lock = true);
        ~FileLock();
        std::string lock_filename;

        // Disable copying and moving
        FileLock(const FileLock&) = delete;
        FileLock& operator=(const FileLock&) = delete;
        FileLock(FileLock&&) = delete;
        FileLock& operator=(FileLock&&) = delete;

    private:
#if defined _WIN32 || defined _WIN64
        void* hFile = (void*)-1;
#else
        int fd = -1;
#endif
    };
}