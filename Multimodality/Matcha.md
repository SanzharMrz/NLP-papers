
use Pix2Struct as the base model and
further pretrain it with chart derendering and math
reasoning tasks.

**In chart derendering**, given a plot/chart,
the image-to-text model is required to generate its
underlying data table or the code used to render
it. 
The second task is **math reasoning pretraining.**
We pick two numerical reasoning dataset MATH
(Saxton et al., 2019) and DROP (Dua et al., 2019),
render the input into images and the image-to-text
model needs to decode the answers.
