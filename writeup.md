## Problem (unicode1)
```
>>> chr(0)
'\x00'
>>> repr(chr(0))
"'\\x00'"
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

(a) `chr(0)` returns the unicode character '\x00', which is 0.
(b) `repr(chr(0))` outputs a readable, escape-sequenced code "'\\x00'", whereas the printed representation is blank.
(c) If it's printed it's invisible, but it's visible when using repr().


## Problem (unicode2)
```
>>> test_string = "hello my name is michael"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello my name is michael'
>>> utf16_encoded = test_string.encode("utf-16")
>>> utf32_encoded = test_string.encode("utf-32")
>>> len(list(utf8_encoded))
24
>>> len(list(utf16_encoded))
50
>>> len(list(utf32_encoded))
100
```
(a) UTF-8 encoded bytes is shorter / more compact.

(b) This treats each byte independently and tries to decode it as a full UTF-8 character. However, UTF-8 is variable-length encoding and one character could map to more than one byte.

```
>>> decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```

(c)  2-byte sequences must start with `110xxxxx`, so the sequence b'\x80\x80' would fail.
```
>>> b'\x80\x80'.decode("utf-8")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Problem (train_bpe_tinystories)

(a) It takes 5 minutes to run on CPU. The longest token in the vocab is b' accomplishment'.

(b) get_pair_to_merge function takes the most time (73% of total). It's called once per merge and each call iterates over all pairs.


# Problem (train_bpe_expts_owt)

(a) Training on 5000 documents took 28 minutes. The longest token in the vocabulary is '—————————————————————-' (length 64). From inspecting the file, seems like many documents are using '—————————————————————-' as visual separator.

(b) 69.6% of TinyStories tokens exist in OpenWebText's vocab, but only 21.7% of OpenWebText's vocab exist in TinyStories. This makes sense since OWT's tokenizer underwent 3.2x more merges so it learned more tokens that TinyStories didn't. And most TinyStories merges OWT also learned.
Both tokenizers peak at 5 bytes, but OWT has a heavier right tail with more tokens up to 13+ bytes.

![token_length_distribution](./plots/token_length_distribution.png)

