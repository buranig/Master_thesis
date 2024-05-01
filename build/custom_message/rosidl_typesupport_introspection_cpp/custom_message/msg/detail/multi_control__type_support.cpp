// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from custom_message:msg/MultiControl.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "custom_message/msg/detail/multi_control__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace custom_message
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void MultiControl_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) custom_message::msg::MultiControl(_init);
}

void MultiControl_fini_function(void * message_memory)
{
  auto typed_message = static_cast<custom_message::msg::MultiControl *>(message_memory);
  typed_message->~MultiControl();
}

size_t size_function__MultiControl__multi_control(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<custom_message::msg::ControlInputs> *>(untyped_member);
  return member->size();
}

const void * get_const_function__MultiControl__multi_control(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<custom_message::msg::ControlInputs> *>(untyped_member);
  return &member[index];
}

void * get_function__MultiControl__multi_control(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<custom_message::msg::ControlInputs> *>(untyped_member);
  return &member[index];
}

void fetch_function__MultiControl__multi_control(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const custom_message::msg::ControlInputs *>(
    get_const_function__MultiControl__multi_control(untyped_member, index));
  auto & value = *reinterpret_cast<custom_message::msg::ControlInputs *>(untyped_value);
  value = item;
}

void assign_function__MultiControl__multi_control(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<custom_message::msg::ControlInputs *>(
    get_function__MultiControl__multi_control(untyped_member, index));
  const auto & value = *reinterpret_cast<const custom_message::msg::ControlInputs *>(untyped_value);
  item = value;
}

void resize_function__MultiControl__multi_control(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<custom_message::msg::ControlInputs> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember MultiControl_message_member_array[1] = {
  {
    "multi_control",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<custom_message::msg::ControlInputs>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_message::msg::MultiControl, multi_control),  // bytes offset in struct
    nullptr,  // default value
    size_function__MultiControl__multi_control,  // size() function pointer
    get_const_function__MultiControl__multi_control,  // get_const(index) function pointer
    get_function__MultiControl__multi_control,  // get(index) function pointer
    fetch_function__MultiControl__multi_control,  // fetch(index, &value) function pointer
    assign_function__MultiControl__multi_control,  // assign(index, value) function pointer
    resize_function__MultiControl__multi_control  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers MultiControl_message_members = {
  "custom_message::msg",  // message namespace
  "MultiControl",  // message name
  1,  // number of fields
  sizeof(custom_message::msg::MultiControl),
  MultiControl_message_member_array,  // message members
  MultiControl_init_function,  // function to initialize message memory (memory has to be allocated)
  MultiControl_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t MultiControl_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &MultiControl_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace custom_message


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<custom_message::msg::MultiControl>()
{
  return &::custom_message::msg::rosidl_typesupport_introspection_cpp::MultiControl_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, custom_message, msg, MultiControl)() {
  return &::custom_message::msg::rosidl_typesupport_introspection_cpp::MultiControl_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
