// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_message:msg/MultiState.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__MULTI_STATE__STRUCT_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__MULTI_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'multiple_state'
#include "custom_message/msg/detail/full_state__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__custom_message__msg__MultiState __attribute__((deprecated))
#else
# define DEPRECATED__custom_message__msg__MultiState __declspec(deprecated)
#endif

namespace custom_message
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MultiState_
{
  using Type = MultiState_<ContainerAllocator>;

  explicit MultiState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit MultiState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _multiple_state_type =
    std::vector<custom_message::msg::FullState_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<custom_message::msg::FullState_<ContainerAllocator>>>;
  _multiple_state_type multiple_state;

  // setters for named parameter idiom
  Type & set__multiple_state(
    const std::vector<custom_message::msg::FullState_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<custom_message::msg::FullState_<ContainerAllocator>>> & _arg)
  {
    this->multiple_state = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_message::msg::MultiState_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_message::msg::MultiState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_message::msg::MultiState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_message::msg::MultiState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_message::msg::MultiState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_message::msg::MultiState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_message::msg::MultiState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_message::msg::MultiState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_message::msg::MultiState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_message::msg::MultiState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_message__msg__MultiState
    std::shared_ptr<custom_message::msg::MultiState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_message__msg__MultiState
    std::shared_ptr<custom_message::msg::MultiState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MultiState_ & other) const
  {
    if (this->multiple_state != other.multiple_state) {
      return false;
    }
    return true;
  }
  bool operator!=(const MultiState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MultiState_

// alias to use template instance with default allocator
using MultiState =
  custom_message::msg::MultiState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_message

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__MULTI_STATE__STRUCT_HPP_
