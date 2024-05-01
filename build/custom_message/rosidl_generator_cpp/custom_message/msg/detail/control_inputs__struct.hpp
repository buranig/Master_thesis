// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_message:msg/ControlInputs.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__CONTROL_INPUTS__STRUCT_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__CONTROL_INPUTS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__custom_message__msg__ControlInputs __attribute__((deprecated))
#else
# define DEPRECATED__custom_message__msg__ControlInputs __declspec(deprecated)
#endif

namespace custom_message
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ControlInputs_
{
  using Type = ControlInputs_<ContainerAllocator>;

  explicit ControlInputs_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->throttle = 0.0f;
      this->delta = 0.0f;
    }
  }

  explicit ControlInputs_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->throttle = 0.0f;
      this->delta = 0.0f;
    }
  }

  // field types and members
  using _throttle_type =
    float;
  _throttle_type throttle;
  using _delta_type =
    float;
  _delta_type delta;

  // setters for named parameter idiom
  Type & set__throttle(
    const float & _arg)
  {
    this->throttle = _arg;
    return *this;
  }
  Type & set__delta(
    const float & _arg)
  {
    this->delta = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_message::msg::ControlInputs_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_message::msg::ControlInputs_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_message::msg::ControlInputs_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_message::msg::ControlInputs_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_message::msg::ControlInputs_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_message::msg::ControlInputs_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_message::msg::ControlInputs_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_message::msg::ControlInputs_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_message::msg::ControlInputs_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_message::msg::ControlInputs_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_message__msg__ControlInputs
    std::shared_ptr<custom_message::msg::ControlInputs_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_message__msg__ControlInputs
    std::shared_ptr<custom_message::msg::ControlInputs_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ControlInputs_ & other) const
  {
    if (this->throttle != other.throttle) {
      return false;
    }
    if (this->delta != other.delta) {
      return false;
    }
    return true;
  }
  bool operator!=(const ControlInputs_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ControlInputs_

// alias to use template instance with default allocator
using ControlInputs =
  custom_message::msg::ControlInputs_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_message

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__CONTROL_INPUTS__STRUCT_HPP_
