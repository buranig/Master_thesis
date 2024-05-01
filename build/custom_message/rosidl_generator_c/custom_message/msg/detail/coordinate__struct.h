// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_message:msg/Coordinate.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__COORDINATE__STRUCT_H_
#define CUSTOM_MESSAGE__MSG__DETAIL__COORDINATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Coordinate in the package custom_message.
typedef struct custom_message__msg__Coordinate
{
  float x;
  float y;
} custom_message__msg__Coordinate;

// Struct for a sequence of custom_message__msg__Coordinate.
typedef struct custom_message__msg__Coordinate__Sequence
{
  custom_message__msg__Coordinate * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_message__msg__Coordinate__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__COORDINATE__STRUCT_H_
