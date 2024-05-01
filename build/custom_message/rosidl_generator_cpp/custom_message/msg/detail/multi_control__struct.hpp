// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_message:msg/MultiControl.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__MULTI_CONTROL__STRUCT_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__MULTI_CONTROL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'multi_control'
#include "custom_message/msg/detail/control_inputs__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__custom_message__msg__MultiControl __attribute__((deprecated))
#else
# define DEPRECATED__custom_message__msg__MultiControl __declspec(deprecated)
#endif

namespace custom_message
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MultiControl_
{
  using Type = MultiControl_<ContainerAllocator>;

  explicit MultiControl_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit MultiControl_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _multi_control_type =
    std::vector<custom_message::msg::ControlInputs_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<custom_message::msg::ControlInputs_<ContainerAllocator>>>;
  _multi_control_type multi_control;

  // setters for named parameter idiom
  Type & set__multi_control(
    const std::vector<custom_message::msg::ControlInputs_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<custom_message::msg::ControlInputs_<ContainerAllocator>>> & _arg)
  {
    this->multi_control = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_message::msg::MultiControl_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_message::msg::MultiControl_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_message::msg::MultiControl_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_message::msg::MultiControl_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_message::msg::MultiControl_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_message::msg::MultiControl_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_message::msg::MultiControl_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_message::msg::MultiControl_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_message::msg::MultiControl_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_message::msg::MultiControl_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_message__msg__MultiControl
    std::shared_ptr<custom_message::msg::MultiControl_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_message__msg__MultiControl
    std::shared_ptr<custom_message::msg::MultiControl_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MultiControl_ & other) const
  {
    if (this->multi_control != other.multi_control) {
      return false;
    }
    return true;
  }
  bool operator!=(const MultiControl_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MultiControl_

// alias to use template instance with default allocator
using MultiControl =
  custom_message::msg::MultiControl_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_message

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__MULTI_CONTROL__STRUCT_HPP_
