// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from custom_message:msg/MultiState.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__MULTI_STATE__FUNCTIONS_H_
#define CUSTOM_MESSAGE__MSG__DETAIL__MULTI_STATE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "custom_message/msg/rosidl_generator_c__visibility_control.h"

#include "custom_message/msg/detail/multi_state__struct.h"

/// Initialize msg/MultiState message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * custom_message__msg__MultiState
 * )) before or use
 * custom_message__msg__MultiState__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__MultiState__init(custom_message__msg__MultiState * msg);

/// Finalize msg/MultiState message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
void
custom_message__msg__MultiState__fini(custom_message__msg__MultiState * msg);

/// Create msg/MultiState message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * custom_message__msg__MultiState__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
custom_message__msg__MultiState *
custom_message__msg__MultiState__create();

/// Destroy msg/MultiState message.
/**
 * It calls
 * custom_message__msg__MultiState__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
void
custom_message__msg__MultiState__destroy(custom_message__msg__MultiState * msg);

/// Check for msg/MultiState message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__MultiState__are_equal(const custom_message__msg__MultiState * lhs, const custom_message__msg__MultiState * rhs);

/// Copy a msg/MultiState message.
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
custom_message__msg__MultiState__copy(
  const custom_message__msg__MultiState * input,
  custom_message__msg__MultiState * output);

/// Initialize array of msg/MultiState messages.
/**
 * It allocates the memory for the number of elements and calls
 * custom_message__msg__MultiState__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__MultiState__Sequence__init(custom_message__msg__MultiState__Sequence * array, size_t size);

/// Finalize array of msg/MultiState messages.
/**
 * It calls
 * custom_message__msg__MultiState__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
void
custom_message__msg__MultiState__Sequence__fini(custom_message__msg__MultiState__Sequence * array);

/// Create array of msg/MultiState messages.
/**
 * It allocates the memory for the array and calls
 * custom_message__msg__MultiState__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
custom_message__msg__MultiState__Sequence *
custom_message__msg__MultiState__Sequence__create(size_t size);

/// Destroy array of msg/MultiState messages.
/**
 * It calls
 * custom_message__msg__MultiState__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
void
custom_message__msg__MultiState__Sequence__destroy(custom_message__msg__MultiState__Sequence * array);

/// Check for msg/MultiState message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__MultiState__Sequence__are_equal(const custom_message__msg__MultiState__Sequence * lhs, const custom_message__msg__MultiState__Sequence * rhs);

/// Copy an array of msg/MultiState messages.
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
custom_message__msg__MultiState__Sequence__copy(
  const custom_message__msg__MultiState__Sequence * input,
  custom_message__msg__MultiState__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__MULTI_STATE__FUNCTIONS_H_
