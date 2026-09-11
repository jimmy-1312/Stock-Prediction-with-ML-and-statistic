**How computer store numbers**

Input decimal number -> turn into base 2 -> find the first 1 -> round off to the left nearest t bits(decide by float32 or 64)-> store inside RAM, include:(1)sign digit, (2)precision digit: without the leftmost 1, (3)exponent digit.

**How accurate v.s. How large can a float type store**
```
{
    "float32" : {
        "sign":1 bit,
        "precision":23 bits,
        "exponent":8 bits == 256(-126->127)(-127 for 0, 128 for infinite or Nan)
        }
    "float64" : {
        "sign":1 bit,
        "precision": 52 bits,
        "exponent":11 bits == 2048(-1022->1023)(-1023 for 0, 1024 for infinite or Nan)
        }
        
    }
```