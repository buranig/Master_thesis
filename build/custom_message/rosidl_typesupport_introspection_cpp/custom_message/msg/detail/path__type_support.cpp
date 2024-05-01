// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from custom_message:msg/Path.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "custom_message/msg/detail/path__struct.hpp"
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

void Path_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) custom_message::msg::Path(_init);
}

void Path_fini_function(void * message_memory)
{
  auto typed_message = static_cast<custom_message::msg::Path *>(message_memory);
  typed_message->~Path();
}

size_t size_function__Path__path(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<custom_message::msg::Coordinate> *>(untyped_member);
  return member->size();
}

const void * get_const_function__Path__path(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<custom_message::msg::Coordinate> *>(untyped_member);
  return &member[index];
}

void * get_function__Path__path(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<custom_message::msg::Coordinate> *>(untyped_member);
  return &member[index];
}

void fetch_function__Path__path(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const custom_message::msg::Coordinate *>(
    get_const_function__Path__path(untyped_member, index));
  auto & value = *reinterpret_cast<custom_message::msg::Coordinate *>(untyped_value);
  value = item;
}

void assign_function__Path__path(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<custom_message::msg::Coordinate *>(
    get_function__Path__path(untyped_member, index));
  const auto & value = *reinterpret_cast<const custom_message::msg::Coordinate *>(untyped_value);
  item = value;
}

void resize_function__Path__path(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<custom_message::msg::Coordinate> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember Path_message_member_array[1] = {
  {
    "path",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<custom_message::msg::Coordinate>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_message::msg::Path, path),  // bytes offset in struct
    nullptr,  // default value
    size_function__Path__path,  // size() function pointer
    get_const_function__Path__path,  // get_const(index) function pointer
    get_function__Path__path,  // get(index) function pointer
    fetch_function__Path__path,  // fetch(index, &value) function pointer
    assign_function__Path__path,  // assign(index, value) function pointer
    resize_function__Path__path  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Path_message_members = {
  "custom_message::msg",  // message namespace
  "Path",  // message name
  1,  // number of fields
  sizeof(custom_message::msg::Path),
  Path_message_member_array,  // message members
  Path_init_function,  // function to initialize message memory (memory has to be allocated)
  Path_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t Path_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Path_message_members,
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
get_message_type_support_handle<custom_message::msg::Path>()
{
  return &::custom_message::msg::rosidl_typesupport_introspection_cpp::Path_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, custom_message, msg, Path)() {
  return &::custom_message::msg::rosidl_typesupport_introspection_cpp::Path_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
