// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_message:msg/FullState.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__FULL_STATE__STRUCT_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__FULL_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__custom_message__msg__FullState __attribute__((deprecated))
#else
# define DEPRECATED__custom_message__msg__FullState __declspec(deprecated)
#endif

namespace custom_message
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct FullState_
{
  using Type = FullState_<ContainerAllocator>;

  explicit FullState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x = 0.0;
      this->y = 0.0;
      this->yaw = 0.0;
      this->v = 0.0;
      this->omega = 0.0;
      this->delta = 0.0;
      this->throttle = 0.0;
    }
  }

  explicit FullState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x = 0.0;
      this->y = 0.0;
      this->yaw = 0.0;
      this->v = 0.0;
      this->omega = 0.0;
      this->delta = 0.0;
      this->throttle = 0.0;
    }
  }

  // field types and members
  using _x_type =
    double;
  _x_type x;
  using _y_type =
    double;
  _y_type y;
  using _yaw_type =
    double;
  _yaw_type yaw;
  using _v_type =
    double;
  _v_type v;
  using _omega_type =
    double;
  _omega_type omega;
  using _delta_type =
    double;
  _delta_type delta;
  using _throttle_type =
    double;
  _throttle_type throttle;

  // setters for named parameter idiom
  Type & set__x(
    const double & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const double & _arg)
  {
    this->y = _arg;
    return *this;
  }
  Type & set__yaw(
    const double & _arg)
  {
    this->yaw = _arg;
    return *this;
  }
  Type & set__v(
    const double & _arg)
  {
    this->v = _arg;
    return *this;
  }
  Type & set__omega(
    const double & _arg)
  {
    this->omega = _arg;
    return *this;
  }
  Type & set__delta(
    const double & _arg)
  {
    this->delta = _arg;
    return *this;
  }
  Type & set__throttle(
    const double & _arg)
  {
    this->throttle = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_message::msg::FullState_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_message::msg::FullState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_message::msg::FullState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_message::msg::FullState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_message::msg::FullState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_message::msg::FullState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_message::msg::FullState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_message::msg::FullState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_message::msg::FullState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_message::msg::FullState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_message__msg__FullState
    std::shared_ptr<custom_message::msg::FullState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_message__msg__FullState
    std::shared_ptr<custom_message::msg::FullState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const FullState_ & other) const
  {
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    if (this->yaw != other.yaw) {
      return false;
    }
    if (this->v != other.v) {
      return false;
    }
    if (this->omega != other.omega) {
      return false;
    }
    if (this->delta != other.delta) {
      return false;
    }
    if (this->throttle != other.throttle) {
      return false;
    }
    return true;
  }
  bool operator!=(const FullState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct FullState_

// alias to use template instance with default allocator
using FullState =
  custom_message::msg::FullState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_message

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__FULL_STATE__STRUCT_HPP_
