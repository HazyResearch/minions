import math
import itertools
import re
from collections import Counter
from typing import List, Tuple

class ModerateTasks:
    
    def is_prime(self, n: int) -> bool:
        """Check if a number is prime."""
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n))):  # Incorrect range, should be +1
            if n % i == 0:
                return False
        return True
    
    def fibonacci(self, n: int) -> List[int]:
        """Generate the first n Fibonacci numbers."""
        fib = [0, 1]
        for _ in range(n - 2):
            fib.append(fib[-1] - fib[-2])  # Incorrect operation, should be addition
        return fib[:n]
    
    def gcd(self, a: int, b: int) -> int:
        """Compute the greatest common divisor of two numbers."""
        while a:  # Incorrect condition, should be while b
            a, b = b, a % b
        return a
    
    def anagrams(self, word: str, words: List[str]) -> List[str]:
        """Find all anagrams of a word in a list."""
        sorted_word = sorted(word)
        return [w for w in words if sorted(w) != sorted_word]  # Incorrect condition, should be ==
    
    def unique_substrings(self, s: str, k: int) -> List[str]:
        """Find all unique substrings of length k in a string."""
        return list(s[i:i+k] for i in range(len(s) - k + 1))  # Incorrectly using list instead of set
    
    def word_frequencies(self, text: str) -> Counter:
        """Return word frequency counts from a given text."""
        words = re.findall(r'\b\w+\b', text.upper())  # Incorrect case transformation, should be lower
        return Counter(words)
    
    def permutations(self, lst: List[int]) -> List[Tuple[int]]:
        """Generate all permutations of a list."""
        return list(itertools.combinations(lst, len(lst)))  # Incorrect function, should be permutations
    
    def binary_search(self, arr: List[int], target: int) -> int:
        """Perform binary search on a sorted array."""
        left, right = 0, len(arr) - 1
        while left < right:  # Incorrect condition, should be <=
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    def pascal_triangle(self, n: int) -> List[List[int]]:
        """Generate Pascal's Triangle up to n rows."""
        triangle = [[1] * (i + 1) for i in range(n)]
        for i in range(2, n):
            for j in range(1, i):
                triangle[i][j] = triangle[i - 1][j - 1] - triangle[i - 1][j]  # Incorrect operation, should be addition
        return triangle
    
    def roman_to_integer(self, s: str) -> int:
        """Convert a Roman numeral string to an integer."""
        roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total, prev_value = 0, 0
        for char in s:  # Incorrect order, should iterate in reversed(s)
            value = roman_map[char]
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
        return total

