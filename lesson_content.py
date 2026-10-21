"""Lessons and quiz questions: 4 topics x 3 difficulty levels x 4 questions.

The training domain is "Python Programming".  Each question has:
  q       the question text
  code    (optional) a code snippet shown under the question
  options four answers
  answer  the correct option (text)
  why     a short explanation shown after the quiz
"""

TOPICS = ["Python Basics", "Control Flow", "Functions", "Data Structures"]


def Q(q, options, answer, why, code=None):
    return {"q": q, "options": options, "answer": answer, "why": why, "code": code}


LESSONS = {
    # ------------------------------------------------------------------ Python Basics
    ("Python Basics", "Easy"): {
        "content": """### Variables and data types
A **variable** is a name that stores a value. You create it with `=`:

```python
name = "Ali"      # str   (text)
age = 20          # int   (whole number)
height = 5.9      # float (decimal number)
passed = True     # bool  (True / False)
```
Use `print()` to show a value and `type()` to see its type.
Variable names may contain letters, digits and `_`, cannot start with a digit and cannot be Python keywords.""",
        "questions": [
            Q("Which line creates an integer variable?", ["x = 5", 'x = "5"', "x = 5.0", "x = [5]"], "x = 5",
              "`5` without quotes or a decimal point is an int."),
            Q("What does this print?", ["<class 'int'>", "<class 'float'>", "<class 'str'>", "<class 'double'>"],
              "<class 'float'>", "3.14 is a decimal number, so its type is float.", code="print(type(3.14))"),
            Q("Which is a valid variable name?", ["2name", "my-name", "my_name", "class"], "my_name",
              "Names cannot start with a digit, contain `-`, or be a keyword such as `class`."),
            Q("What is the output?", ["105", "15", "Error", "10 5"], "105",
              "Both values are strings, so `+` joins them (concatenation).", code='print("10" + "5")'),
        ],
    },
    ("Python Basics", "Medium"): {
        "content": """### Operators, f-strings and conversion
| Operator | Meaning | Example |
|---|---|---|
| `/` | true division | `7 / 2` -> `3.5` |
| `//` | floor division | `7 // 2` -> `3` |
| `%` | remainder | `7 % 3` -> `1` |
| `**` | power | `2 ** 3` -> `8` |

**f-strings** put values inside text: `f"x is {x}"`.
Convert types with `int("7")`, `float("2.5")`, `str(10)`.""",
        "questions": [
            Q("What is the result?", ["3.5", "3", "4", "3.0"], "3", "`//` divides and rounds down to a whole number.",
              code="print(7 // 2)"),
            Q("What is the result?", ["2", "1", "0", "3"], "1", "7 divided by 3 leaves a remainder of 1.",
              code="print(7 % 3)"),
            Q("What is the result?", ["6", "8", "9", "5"], "8", "`**` is the power operator: 2 x 2 x 2.",
              code="print(2 ** 3)"),
            Q("What does this print?", ["x is {x}", "x is 4", 'f"x is 4"', "Error"], "x is 4",
              "The f-string replaces `{x}` with the value of x.", code='x = 4\nprint(f"x is {x}")'),
        ],
    },
    ("Python Basics", "Hard"): {
        "content": """### Precedence and type pitfalls
* Order of operations: `**` first, then `* / // %`, then `+ -`.
* Dividing with `/` always gives a **float**.
* `int("3.7")` raises a `ValueError`; convert with `int(float("3.7"))`.
* Multiplying a string repeats it: `"ab" * 2` -> `"abab"`.
* Any non-empty string is **truthy**, even `"False"`.""",
        "questions": [
            Q("What is printed?", ["14", "14.0", "20.0", "26.0"], "14.0",
              "4**2 = 16, 3*16 = 48, 48/4 = 12.0, 2 + 12.0 = 14.0.", code="print(2 + 3 * 4 ** 2 / 4)"),
            Q("What happens?", ["3", "4", "ValueError", "3.7"], "ValueError",
              "`int()` cannot parse a string that contains a decimal point.", code='print(int("3.7"))'),
            Q("What is printed?", ["15", "555", "5 5 5", "Error"], "555", "A string times an integer repeats the string.",
              code='a = "5"\nb = a * 3\nprint(b)'),
            Q("What is printed?", ["False", "True", "Error", "None"], "True",
              "Non-empty strings are truthy, whatever their text says.", code='print(bool("False"))'),
        ],
    },
    # ------------------------------------------------------------------ Control Flow
    ("Control Flow", "Easy"): {
        "content": """### Decisions and loops
```python
if marks >= 50:
    print("Pass")
elif marks >= 40:
    print("Borderline")
else:
    print("Fail")

for i in range(3):      # i = 0, 1, 2
    print(i)

while count > 0:        # repeats while the condition is True
    count -= 1
```
`==` compares values, `=` assigns. `break` leaves a loop early.""",
        "questions": [
            Q("Which keyword starts a condition?", ["if", "when", "check", "case"], "if", "Conditions begin with `if`."),
            Q("What does this print?", ["1 2 3", "0 1 2", "0 1 2 3", "1 2"], "0 1 2",
              "`range(3)` produces 0, 1 and 2.", code="for i in range(3):\n    print(i)"),
            Q("Which operator checks equality?", ["==", "=", "!=", "=>"], "==", "`=` assigns, `==` compares."),
            Q("Which keyword leaves a loop early?", ["break", "stop", "exit", "end"], "break", "`break` ends the loop."),
        ],
    },
    ("Control Flow", "Medium"): {
        "content": """### Loop details
* `range(start, stop, step)` stops **before** `stop`: `range(2, 10, 3)` -> 2, 5, 8.
* `and`, `or`, `not` combine conditions.
* `continue` skips the rest of the current round and goes to the next one.
* A `while` loop must change something inside it, otherwise it never ends.""",
        "questions": [
            Q("What is printed?", ["2 5 8", "2 5 8 10", "3 6 9", "2 4 6 8"], "2 5 8",
              "Start at 2 and add 3 each time while the value is below 10.",
              code='for i in range(2, 10, 3):\n    print(i, end=" ")'),
            Q("What is printed?", ["A", "B", "Nothing", "Error"], "A", "7 is greater than 5 and less than 10.",
              code='x = 7\nif x > 5 and x < 10:\n    print("A")\nelse:\n    print("B")'),
            Q("What does `continue` do?",
              ["Stops the whole loop", "Skips the rest of this round and starts the next one",
               "Exits the program", "Restarts the loop from the first item"],
              "Skips the rest of this round and starts the next one", "`continue` jumps to the next iteration."),
            Q("How many times does the loop body run?", ["2", "3", "4", "Forever"], "3",
              "i becomes 1, 2, 3 and then 3 < 3 is False.", code="i = 0\nwhile i < 3:\n    i += 1"),
        ],
    },
    ("Control Flow", "Hard"): {
        "content": """### Tricky control flow
* Nested loops multiply: an outer loop of 3 and an inner loop of 2 run the inner body 6 times.
* `for ... else`: the `else` block runs only if the loop finished **without** `break`.
* Python allows chained comparisons: `5 > 3 > 1` means `5 > 3 and 3 > 1`.
* `%` is handy for odd/even checks: `i % 2 == 0` is True for even numbers.""",
        "questions": [
            Q("What is printed?", ["4", "6", "10", "3"], "4", "Only the odd numbers 1 and 3 are added.",
              code="total = 0\nfor i in range(1, 5):\n    if i % 2 == 0:\n        continue\n    total += i\nprint(total)"),
            Q("What is printed?", ["done", "Nothing", "Error", "0 1 2"], "done",
              "The loop never hits `break`, so the `else` block runs.",
              code='for i in range(3):\n    if i == 5:\n        break\nelse:\n    print("done")'),
            Q("What is printed?", ["5", "6", "3", "2"], "6", "3 outer rounds x 2 inner rounds = 6.",
              code="count = 0\nfor a in range(3):\n    for b in range(2):\n        count += 1\nprint(count)"),
            Q("What is printed?", ["True", "False", "Error", "None"], "True", "Both 5 > 3 and 3 > 1 are True.",
              code="print(5 > 3 > 1)"),
        ],
    },
    # ------------------------------------------------------------------ Functions
    ("Functions", "Easy"): {
        "content": """### Writing functions
```python
def add(a, b):
    return a + b

result = add(2, 3)   # call the function
print(result)        # 5
```
`def` defines a function, parameters go in brackets and `return` sends a value back to the caller.""",
        "questions": [
            Q("Which keyword defines a function?", ["def", "func", "function", "define"], "def", "Functions start with `def`."),
            Q("What is printed?", ["5", "23", "Error", "None"], "5", "The function returns 2 + 3.",
              code="def add(a, b):\n    return a + b\nprint(add(2, 3))"),
            Q("What does `return` do?",
              ["Prints a value", "Sends a value back to the caller and ends the function",
               "Starts a loop", "Imports a module"],
              "Sends a value back to the caller and ends the function", "`return` hands back the result."),
            Q("Which line calls the function `hello`?", ["hello()", "def hello", "call hello", "hello:"], "hello()",
              "A call uses the name followed by brackets."),
        ],
    },
    ("Functions", "Medium"): {
        "content": """### Parameters and scope
* **Default values**: `def power(x, n=2)` uses `n = 2` when no value is given.
* **Keyword arguments**: `f(b=1, a=5)` pass values by name, in any order.
* A function without `return` gives back `None`.
* Variables created inside a function are **local**; they disappear when it ends.""",
        "questions": [
            Q("What is printed?", ["6", "9", "3", "27"], "9", "`n` defaults to 2, so 3 ** 2 = 9.",
              code="def power(x, n=2):\n    return x ** n\nprint(power(3))"),
            Q("What is the output?", ["None", "hi", "hi, then None", "Error"], "hi, then None",
              "`f()` prints hi but returns None, which is then printed.",
              code='def f():\n    print("hi")\nx = f()\nprint(x)'),
            Q("What is printed?", ["4", "-4", "6", "Error"], "4", "a = 5 and b = 1, so 5 - 1 = 4.",
              code="def f(a, b):\n    return a - b\nprint(f(b=1, a=5))"),
            Q("Variables created inside a function are...", ["Global", "Local to the function", "Constants", "Shared with all files"],
              "Local to the function", "They exist only while the function runs."),
        ],
    },
    ("Functions", "Hard"): {
        "content": """### Advanced functions
* **Mutable default argument trap**: a default list is created **once** and reused between calls.
* `*args` collects extra positional arguments into a tuple.
* **Recursion**: a function that calls itself needs a base case.
* Assigning to a name inside a function creates a new local variable; it does not change a global one unless you use `global`.""",
        "questions": [
            Q("What is printed?", ["[2]", "[1, 2]", "[1]", "Error"], "[1, 2]",
              "The default list is shared between calls, so it keeps the 1.",
              code="def add(item, lst=[]):\n    lst.append(item)\n    return lst\nadd(1)\nprint(add(2))"),
            Q("What is printed?", ["6", "(1, 2, 3)", "Error", "3"], "6", "`*nums` collects 1, 2, 3 and `sum` adds them.",
              code="def total(*nums):\n    return sum(nums)\nprint(total(1, 2, 3))"),
            Q("What is printed?", ["10", "24", "12", "16"], "24", "4 x 3 x 2 x 1 = 24.",
              code="def f(n):\n    return 1 if n <= 1 else n * f(n - 1)\nprint(f(4))"),
            Q("What is printed?", ["5", "10", "Error", "None"], "10", "The `x = 5` inside the function is a separate local variable.",
              code="x = 10\ndef change():\n    x = 5\nchange()\nprint(x)"),
        ],
    },
    # ------------------------------------------------------------------ Data Structures
    ("Data Structures", "Easy"): {
        "content": """### Lists and dictionaries
```python
fruits = ["apple", "banana", "cherry"]   # list: ordered, changeable
print(fruits[0])      # apple  (index starts at 0)
print(len(fruits))    # 3

marks = {"Ali": 80, "Sara": 92}          # dict: key -> value
print(marks["Sara"])  # 92
```
Other types: **tuple** `(1, 2)` (cannot change) and **set** `{1, 2}` (no duplicates).""",
        "questions": [
            Q("What is printed?", ["2", "3", "4", "Error"], "3", "The list has three items.", code="nums = [1, 2, 3]\nprint(len(nums))"),
            Q("Which brackets create a list?", ["[ ]", "( )", "{ }", "< >"], "[ ]", "Lists use square brackets."),
            Q("What is printed?", ["a", "b", "c", "0"], "a", "Indexing starts at 0.", code='fruits = ["a", "b", "c"]\nprint(fruits[0])'),
            Q("Which type stores key-value pairs?", ["list", "tuple", "dict", "set"], "dict", "A dictionary maps keys to values."),
        ],
    },
    ("Data Structures", "Medium"): {
        "content": """### Working with collections
* `list.append(x)` adds to the end; `list.remove(x)` deletes a value.
* **Slicing**: `items[1:4]` gives positions 1, 2, 3 (the end is excluded).
* `d["key"]` reads a value; `d.get("key", default)` is safe if the key may be missing.
* A `set` removes duplicates: `len(set([1, 1, 2]))` is 2.""",
        "questions": [
            Q("What is printed?", ["[1, 2, 3]", "[1, 2, 3, 4]", "[4, 1, 2, 3]", "Error"], "[1, 2, 3, 4]",
              "`append` adds the value at the end.", code="nums = [1, 2, 3]\nnums.append(4)\nprint(nums)"),
            Q("What is printed?", ["1", "2", '"b"', "Error"], "2", 'The key "b" has the value 2.',
              code='d = {"a": 1, "b": 2}\nprint(d["b"])'),
            Q("What is printed?", ["[1, 2, 3, 4]", "[2, 3, 4]", "[2, 3, 4, 5]", "[1, 2, 3]"], "[2, 3, 4]",
              "Positions 1 to 3; the end index 4 is excluded.", code="print([1, 2, 3, 4, 5][1:4])"),
            Q("What is printed?", ["5", "3", "2", "Error"], "3", "The set keeps only 1, 2 and 3.",
              code="print(len(set([1, 1, 2, 3, 3])))"),
        ],
    },
    ("Data Structures", "Hard"): {
        "content": """### Deeper data-structure ideas
* **List comprehension**: `[x * x for x in range(5) if x % 2 == 0]` builds a list in one line.
* **Tuples are immutable**: assigning to `t[0]` raises `TypeError`.
* **Aliasing**: `b = a` does not copy a list, both names point to the same list.
* `dict.get(key, default)` returns the default instead of raising `KeyError`.""",
        "questions": [
            Q("What is printed?", ["[0, 4, 16]", "[0, 1, 4, 9, 16]", "[1, 9]", "[0, 2, 4]"], "[0, 4, 16]",
              "Only the even numbers 0, 2, 4 are squared.", code="print([x * x for x in range(5) if x % 2 == 0])"),
            Q("What happens?", ["t becomes (9, 2, 3)", "TypeError", "Nothing changes, no error", "ValueError"], "TypeError",
              "Tuples cannot be changed after creation.", code="t = (1, 2, 3)\nt[0] = 9"),
            Q("What is printed?", ["[1, 2, 3]", "[1, 2, 3, 4]", "[4]", "Error"], "[1, 2, 3, 4]",
              "`b` and `a` are the same list.", code="a = [1, 2, 3]\nb = a\nb.append(4)\nprint(a)"),
            Q("What is printed?", ["None", "0", "KeyError", '"z"'], "0", "`get` returns the default 0 when the key is missing.",
              code='d = {"a": 1}\nprint(d.get("z", 0))'),
        ],
    },
}

# quick self-check when the module is imported
for _k, _v in LESSONS.items():
    for _q in _v["questions"]:
        assert _q["answer"] in _q["options"], (_k, _q["q"])
