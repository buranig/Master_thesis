// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_message:msg/MultiplePaths.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__STRUCT_H_
#define CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'multiple_path'
#include "custom_message/msg/detail/path__struct.h"

/// Struct defined in msg/MultiplePaths in the package custom_message.
typedef struct custom_message__msg__MultiplePaths
{
  custom_message__msg__Path__Sequence multiple_path;
} custom_message__msg__MultiplePaths;

// Struct for a sequence of custom_message__msg__MultiplePaths.
typedef struct custom_message__msg__MultiplePaths__Sequence
{
  custom_message__msg__MultiplePaths * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_message__msg__MultiplePaths__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__STRUCT_H_
