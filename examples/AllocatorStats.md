# Allocator Statistics

TensorFlow Large Model Support provides APIs to retrieve statistics from
TensorFlow's GPU memory allocator. This document lists the APIs and what
they provide.


## General allocator statistics

```python
tf.experimental.get_num_allocs(gpu_id)
```
Returns the number of allocation requests made to the allocator.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.


```python
tf.experimental.get_bytes_in_use(gpu_id)
```
Returns the number of bytes in use.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.


```python
tf.experimental.get_peak_bytes_in_use(gpu_id)
```
Returns the peak number of bytes in use.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.


```python
tf.experimental.get_largest_alloc_size(gpu_id)
```
Returns the size in bytes of the largest allocation.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.


```python
tf.experimental.get_bytes_limit(gpu_id)
```
Returns the limit (in bytes) of user allocatable memory.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.


```python
tf.experimental.get_bytes_reserved(gpu_id)
```
Returns the number of bytes of reserved memory.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.

```python
tf.experimental.get_peak_bytes_reserved(gpu_id)
```
Returns the peak number of bytes of reserved memory.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.

```python
tf.experimental.get_bytes_reservable_limit(gpu_id)
```
Returns the limit of reservable memory.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.

## Large Model Support Specific Statistics
The Large Model Support specific statistics provide information about Large
Model Support's memory management. The statics use the following terms:

* active bytes - Active bytes are those bytes consumed by active tensors.
Active tensors are those which are inputs or outputs to currently
executing operations or operations that are being prepared for execution.
* inactive bytes - Inactive bytes are those bytes consumed by inactive tensors.
Inactive tensors are those tensors which are not currently being used by an
executing operation or a soon-to-be executing operation.
* reclaim bytes - Reclaimed bytes are the bytes of inactive tensors which have
been moved from GPU memory to the system (host) memory.
* defragmentation - A method of producing contiguous memory blocks by moving
active bytes to allow free memory blocks between the active bytes to coalesce
into larger contiguous blocks.


```python
tf.experimental.get_bytes_inactive(gpu_id)
```
Returns the number of inactive bytes.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.


```python
tf.experimental.get_bytes_active(gpu_id)
```
Returns the number of active bytes.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.

```python
tf.experimental.get_peak_bytes_active(gpu_id)
```
Returns the peak number of active bytes.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.

```python
tf.experimental.get_bytes_reclaimed(gpu_id)
```
Returns the number of reclaimed bytes.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.

```python
tf.experimental.get_num_single_reclaims(gpu_id)
```
Large Model Support will reclaim the bytes of single tensors when possible.
This returns the number of times single tensors' bytes were reclaimed.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.

```python
tf.experimental.get_num_full_reclaims(gpu_id)
```
When no single tensor reclamation is able to free enough GPU memory for the
allocation request, all tensors are reclaimed. This returns the number
of times all tensors were reclaimed.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.


```python
tf.experimental.get_num_defragmentations(gpu_id)
```
GPU memory may become fragmented such that there are no contiguous blocks which
can fulfill an allocation request, even after reclaiming all inactive
tensors. In this case, active tensors may be moved to allow free blocks to be
coalesced to produce a contiguous memory block large enough to fulfill the
allocation request. The defragmentation function of Large Model Support is
disabled by default. This API returns the number of times defragmentation was
performed.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.


```python
tf.experimental.get_bytes_defragged(gpu_id)
```
The number of bytes moved during GPU memory defragmentation.

**Parameter:** `gpu_id`: The zero indexed GPU ID for which to retrieve the statistic.
