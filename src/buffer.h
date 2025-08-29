#pragma once

#include <cstddef>
#include <stdexcept>

namespace janus
{
    enum class Space
    {
        Host,
        Device
    };

    class BufferException : public std::runtime_error
    {
    public:
        explicit BufferException(const std::string &message) : std::runtime_error(message) {}
    };

    // Forward declaration of template class
    template <typename T, Space S>
    class Buffer;

    // Host specialization
    template <typename T>
    class Buffer<T, Space::Host>
    {
    public:
        Buffer(size_t count);
        ~Buffer();

        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;
        Buffer(Buffer &&other) noexcept;
        Buffer &operator=(Buffer &&other) noexcept;

        T *data() noexcept { return data_; }
        const T *data() const noexcept { return data_; }
        size_t size() const noexcept { return size_; }
        size_t bytes() const noexcept { return size_ * sizeof(T); }

        template <Space O>
        void copy_from(const Buffer<T, O> &other);

        void fill(const T &value);
        void zero();

    private:
        T *data_;
        size_t size_;
    };

    // Device specialization
    template <typename T>
    class Buffer<T, Space::Device>
    {
    public:
        Buffer(size_t count);
        ~Buffer();

        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;
        Buffer(Buffer &&other) noexcept;
        Buffer &operator=(Buffer &&other) noexcept;

        T *data() noexcept { return data_; }
        const T *data() const noexcept { return data_; }
        size_t size() const noexcept { return size_; }
        size_t bytes() const noexcept { return size_ * sizeof(T); }

        template <Space O>
        void copy_from(const Buffer<T, O> &other);

        void fill(const T &value);
        void zero();

    private:
        T *data_;
        size_t size_;
    };
}
