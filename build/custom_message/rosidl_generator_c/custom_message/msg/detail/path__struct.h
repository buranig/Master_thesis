// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_message:msg/Path.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__PATH__STRUCT_H_
#define CUSTOM_MESSAGE__MSG__DETAIL__PATH__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'path'
#include "custom_message/msg/detail/coordinate__struct.h"

/// Struct defined in msg/Path in the package custom_message.
typedef struct custom_message__msg__Path
{
  custom_message__msg__Coordinate__Sequence path;
} custom_message__msg__Path;

// Struct for a sequence of custom_message__msg__Path.
typedef struct custom_message__msg__Path__Sequence
{
  custom_message__msg__Path * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_message__msg__Path__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__PATH__STRUCT_H_
