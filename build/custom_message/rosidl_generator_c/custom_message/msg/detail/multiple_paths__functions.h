// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from custom_message:msg/MultiplePaths.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__FUNCTIONS_H_
#define CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "custom_message/msg/rosidl_generator_c__visibility_control.h"

#include "custom_message/msg/detail/multiple_paths__struct.h"

/// Initialize msg/MultiplePaths message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * custom_message__msg__MultiplePaths
 * )) before or use
 * custom_message__msg__MultiplePaths__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__MultiplePaths__init(custom_message__msg__MultiplePaths * msg);

/// Finalize msg/MultiplePaths message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
void
custom_message__msg__MultiplePaths__fini(custom_message__msg__MultiplePaths * msg);

/// Create msg/MultiplePaths message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * custom_message__msg__MultiplePaths__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
custom_message__msg__MultiplePaths *
custom_message__msg__MultiplePaths__create();

/// Destroy msg/MultiplePaths message.
/**
 * It calls
 * custom_message__msg__MultiplePaths__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
void
custom_message__msg__MultiplePaths__destroy(custom_message__msg__MultiplePaths * msg);

/// Check for msg/MultiplePaths message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__MultiplePaths__are_equal(const custom_message__msg__MultiplePaths * lhs, const custom_message__msg__MultiplePaths * rhs);

/// Copy a msg/MultiplePaths message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__MultiplePaths__copy(
  const custom_message__msg__MultiplePaths * input,
  custom_message__msg__MultiplePaths * output);

/// Initialize array of msg/MultiplePaths messages.
/**
 * It allocates the memory for the number of elements and calls
 * custom_message__msg__MultiplePaths__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__MultiplePaths__Sequence__init(custom_message__msg__MultiplePaths__Sequence * array, size_t size);

/// Finalize array of msg/MultiplePaths messages.
/**
 * It calls
 * custom_message__msg__MultiplePaths__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
void
custom_message__msg__MultiplePaths__Sequence__fini(custom_message__msg__MultiplePaths__Sequence * array);

/// Create array of msg/MultiplePaths messages.
/**
 * It allocates the memory for the array and calls
 * custom_message__msg__MultiplePaths__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
custom_message__msg__MultiplePaths__Sequence *
custom_message__msg__MultiplePaths__Sequence__create(size_t size);

/// Destroy array of msg/MultiplePaths messages.
/**
 * It calls
 * custom_message__msg__MultiplePaths__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
void
custom_message__msg__MultiplePaths__Sequence__destroy(custom_message__msg__MultiplePaths__Sequence * array);

/// Check for msg/MultiplePaths message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__MultiplePaths__Sequence__are_equal(const custom_message__msg__MultiplePaths__Sequence * lhs, const custom_message__msg__MultiplePaths__Sequence * rhs);

/// Copy an array of msg/MultiplePaths messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__MultiplePaths__Sequence__copy(
  const custom_message__msg__MultiplePaths__Sequence * input,
  custom_message__msg__MultiplePaths__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__FUNCTIONS_H_
