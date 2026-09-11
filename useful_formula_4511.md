When need calculate Macaulay durations, it's often convienient to apply the following formula:

Present value = Q/i * (a' i|n - n*(1+i)^-n)
Future value = Q/i * (s' i|n - n)

Q = Fr*T
Derived from the equation use for increasing coupon every period with Q, where start at P at time = 1,
So in Macaulay case we just treat the increasing value as the time interval * the fixed coupon value

Future value = P * a i|n + Q/i * (s i|n - n)

Noted that a' i|n = a i|n * (1+i) = the intermediate present value(start from t = 0)
a i|n = (1 - (1+i)^-n)/i


