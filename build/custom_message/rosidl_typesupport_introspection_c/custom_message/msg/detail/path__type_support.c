// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from custom_message:msg/Path.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "custom_message/msg/detail/path__rosidl_typesupport_introspection_c.h"
#include "custom_message/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "custom_message/msg/detail/path__functions.h"
#include "custom_message/msg/detail/path__struct.h"


// Include directives for member types
// Member `path`
#include "custom_message/msg/coordinate.h"
// Member `path`
#include "custom_message/msg/detail/coordinate__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void custom_message__msg__Path__rosidl_typesupport_introspection_c__Path_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  custom_message__msg__Path__init(message_memory);
}

void custom_message__msg__Path__rosidl_typesupport_introspection_c__Path_fini_function(void * message_memory)
{
  custom_message__msg__Path__fini(message_memory);
}

size_t custom_message__msg__Path__rosidl_typesupport_introspection_c__size_function__Path__path(
  const void * untyped_member)
{
  const custom_message__msg__Coordinate__Sequence * member =
    (const custom_message__msg__Coordinate__Sequence *)(untyped_member);
  return member->size;
}

const void * custom_message__msg__Path__rosidl_typesupport_introspection_c__get_const_function__Path__path(
  const void * untyped_member, size_t index)
{
  const custom_message__msg__Coordinate__Sequence * member =
    (const custom_message__msg__Coordinate__Sequence *)(untyped_member);
  return &member->data[index];
}

void * custom_message__msg__Path__rosidl_typesupport_introspection_c__get_function__Path__path(
  void * untyped_member, size_t index)
{
  custom_message__msg__Coordinate__Sequence * member =
    (custom_message__msg__Coordinate__Sequence *)(untyped_member);
  return &member->data[index];
}

void custom_message__msg__Path__rosidl_typesupport_introspection_c__fetch_function__Path__path(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const custom_message__msg__Coordinate * item =
    ((const custom_message__msg__Coordinate *)
    custom_message__msg__Path__rosidl_typesupport_introspection_c__get_const_function__Path__path(untyped_member, index));
  custom_message__msg__Coordinate * value =
    (custom_message__msg__Coordinate *)(untyped_value);
  *value = *item;
}

void custom_message__msg__Path__rosidl_typesupport_introspection_c__assign_function__Path__path(
  void * untyped_member, size_t index, const void * untyped_value)
{
  custom_message__msg__Coordinate * item =
    ((custom_message__msg__Coordinate *)
    custom_message__msg__Path__rosidl_typesupport_introspection_c__get_function__Path__path(untyped_member, index));
  const custom_message__msg__Coordinate * value =
    (const custom_message__msg__Coordinate *)(untyped_value);
  *item = *value;
}

bool custom_message__msg__Path__rosidl_typesupport_introspection_c__resize_function__Path__path(
  void * untyped_member, size_t size)
{
  custom_message__msg__Coordinate__Sequence * member =
    (custom_message__msg__Coordinate__Sequence *)(untyped_member);
  custom_message__msg__Coordinate__Sequence__fini(member);
  return custom_message__msg__Coordinate__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember custom_message__msg__Path__rosidl_typesupport_introspection_c__Path_message_member_array[1] = {
  {
    "path",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_message__msg__Path, path),  // bytes offset in struct
    NULL,  // default value
    custom_message__msg__Path__rosidl_typesupport_introspection_c__size_function__Path__path,  // size() function pointer
    custom_message__msg__Path__rosidl_typesupport_introspection_c__get_const_function__Path__path,  // get_const(index) function pointer
    custom_message__msg__Path__rosidl_typesupport_introspection_c__get_function__Path__path,  // get(index) function pointer
    custom_message__msg__Path__rosidl_typesupport_introspection_c__fetch_function__Path__path,  // fetch(index, &value) function pointer
    custom_message__msg__Path__rosidl_typesupport_introspection_c__assign_function__Path__path,  // assign(index, value) function pointer
    custom_message__msg__Path__rosidl_typesupport_introspection_c__resize_function__Path__path  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers custom_message__msg__Path__rosidl_typesupport_introspection_c__Path_message_members = {
  "custom_message__msg",  // message namespace
  "Path",  // message name
  1,  // number of fields
  sizeof(custom_message__msg__Path),
  custom_message__msg__Path__rosidl_typesupport_introspection_c__Path_message_member_array,  // message members
  custom_message__msg__Path__rosidl_typesupport_introspection_c__Path_init_function,  // function to initialize message memory (memory has to be allocated)
  custom_message__msg__Path__rosidl_typesupport_introspection_c__Path_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t custom_message__msg__Path__rosidl_typesupport_introspection_c__Path_message_type_support_handle = {
  0,
  &custom_message__msg__Path__rosidl_typesupport_introspection_c__Path_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_custom_message
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_message, msg, Path)() {
  custom_message__msg__Path__rosidl_typesupport_introspection_c__Path_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_message, msg, Coordinate)();
  if (!custom_message__msg__Path__rosidl_typesupport_introspection_c__Path_message_type_support_handle.typesupport_identifier) {
    custom_message__msg__Path__rosidl_typesupport_introspection_c__Path_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &custom_message__msg__Path__rosidl_typesupport_introspection_c__Path_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
