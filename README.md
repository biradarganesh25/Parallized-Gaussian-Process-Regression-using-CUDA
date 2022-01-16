Parallized different steps involved such as LU decomposition, matrix multiplication and achieved a peak speed of over 200 for the decompose function

![](project_explained.png?raw=true "Explaination of project")

The aim of the project was to calculate f(x,y) for a new x,y from observations of other points in the same domain. The hardest part is parallizing equation 3, which needs parallized LU decompostion to solve for f-star. 

The LU parallization is explained in this image: 

![](decompose_explained.jpeg?raw=true "Parallized LU decomposition")

Parallizing other pieces is straightforward.

Note: The commit history is not proper here as the original local repo I was using was on a server and it crashed. This folder was the regular backup I was taking.