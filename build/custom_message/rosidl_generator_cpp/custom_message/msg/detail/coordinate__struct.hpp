// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_message:msg/Coordinate.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__COORDINATE__STRUCT_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__COORDINATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__custom_message__msg__Coordinate __attribute__((deprecated))
#else
# define DEPRECATED__custom_message__msg__Coordinate __declspec(deprecated)
#endif

namespace custom_message
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Coordinate_
{
  using Type = Coordinate_<ContainerAllocator>;

  explicit Coordinate_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x = 0l;
      this->y = 0l;
    }
  }

  explicit Coordinate_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x = 0l;
      this->y = 0l;
    }
  }

  // field types and members
  using _x_type =
    int32_t;
  _x_type x;
  using _y_type =
    int32_t;
  _y_type y;

  // setters for named parameter idiom
  Type & set__x(
    const int32_t & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const int32_t & _arg)
  {
    this->y = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_message::msg::Coordinate_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_message::msg::Coordinate_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_message::msg::Coordinate_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_message::msg::Coordinate_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_message::msg::Coordinate_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_message::msg::Coordinate_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_message::msg::Coordinate_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_message::msg::Coordinate_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_message::msg::Coordinate_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_message::msg::Coordinate_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_message__msg__Coordinate
    std::shared_ptr<custom_message::msg::Coordinate_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_message__msg__Coordinate
    std::shared_ptr<custom_message::msg::Coordinate_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Coordinate_ & other) const
  {
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    return true;
  }
  bool operator!=(const Coordinate_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Coordinate_

// alias to use template instance with default allocator
using Coordinate =
  custom_message::msg::Coordinate_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_message

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__COORDINATE__STRUCT_HPP_
