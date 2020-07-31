def top_k(data, k, axis=1, target=0, target_only=False, desc=True):
    def __push(item, data, axis, desc):
        for idx in range(len(data) - 1, -1, -1):
            if desc is True and data[idx][axis] >= item[axis] or desc is False and data[idx][axis] <= item[axis]:
                result = data[:idx + 1]
                result.append(item)
                result.extend(data[idx + 1:])
                return result
        data.insert(0, item)
        return data

    result = []

    for item in data:
        if len(result) < k or desc is True and item[axis] > result[-1][axis] or desc is False and item[axis] < \
                result[-1][axis]:
            result = __push(item, result, axis=axis, desc=desc)
        if len(result) > k:
            result = result[:k]

    return [i[target] for i in result] if target_only else result