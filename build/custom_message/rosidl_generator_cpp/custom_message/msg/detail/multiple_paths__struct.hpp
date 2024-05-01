// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_message:msg/MultiplePaths.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__STRUCT_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'multiple_path'
#include "custom_message/msg/detail/path__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__custom_message__msg__MultiplePaths __attribute__((deprecated))
#else
# define DEPRECATED__custom_message__msg__MultiplePaths __declspec(deprecated)
#endif

namespace custom_message
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MultiplePaths_
{
  using Type = MultiplePaths_<ContainerAllocator>;

  explicit MultiplePaths_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit MultiplePaths_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _multiple_path_type =
    std::vector<custom_message::msg::Path_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<custom_message::msg::Path_<ContainerAllocator>>>;
  _multiple_path_type multiple_path;

  // setters for named parameter idiom
  Type & set__multiple_path(
    const std::vector<custom_message::msg::Path_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<custom_message::msg::Path_<ContainerAllocator>>> & _arg)
  {
    this->multiple_path = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_message::msg::MultiplePaths_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_message::msg::MultiplePaths_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_message::msg::MultiplePaths_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_message::msg::MultiplePaths_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_message::msg::MultiplePaths_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_message::msg::MultiplePaths_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_message::msg::MultiplePaths_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_message::msg::MultiplePaths_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_message::msg::MultiplePaths_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_message::msg::MultiplePaths_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_message__msg__MultiplePaths
    std::shared_ptr<custom_message::msg::MultiplePaths_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_message__msg__MultiplePaths
    std::shared_ptr<custom_message::msg::MultiplePaths_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MultiplePaths_ & other) const
  {
    if (this->multiple_path != other.multiple_path) {
      return false;
    }
    return true;
  }
  bool operator!=(const MultiplePaths_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MultiplePaths_

// alias to use template instance with default allocator
using MultiplePaths =
  custom_message::msg::MultiplePaths_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_message

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__STRUCT_HPP_
