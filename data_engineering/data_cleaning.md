should i do denoising? and do the training and testing all on the denoised data? 
but that will make the assumptions of that denoised data is accurate and very near to true distribution.
Or should i just another model just for denoising?

Or actually prediction model is already acting like denoising? Yes to the output, but if the input is also noised?

*potential harmful feature values:*
# fatal -> null, NaN, empty
# Critical -> relatively extreme/small, zero(when inputted to log return)
# Moderate -> collinearity, point outliers in feature/residual, violation of statistic assumpytion(resdiual constant variance, mean 0, linearity, normal distribtion) *not confirmed*
