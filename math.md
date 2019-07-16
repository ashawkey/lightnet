# Back Propagation Algorithm

### Fully Connected Layer

##### Single Instance

$x:[1, n],\ y:[1, m],\ W: [n,m],\ b:[1,m]$ （数据为行向量）

$L$ 为最终的标量损失函数值。

$\frac {\partial L} {\partial y}: [1,m]$ 为下一层传来的已知的梯度。

$\frac {\partial y} {\partial W}: [m, nm]$ 需要计算，为方便视W为$[1, mn]$向量。
$$
y = xW + b \\
\frac {\partial L} {\partial W} = \frac {\partial L} {\partial y} \frac {\partial y} {\partial W} \\
y_i = \sum_jx_jW_{ji} \\
\frac {\partial y_i} {\partial W_{jk}} = \begin{cases}0, &i\ne k\\x_i,&i=k\end{cases} \\
\frac {\partial y} {\partial W} = \begin{bmatrix} 
\frac {\partial y_1} {\partial W_{11}} & \frac {\partial y_1} {\partial W_{21}} &\cdots&\frac {\partial y_1} {\partial W_{nm}} \\
\frac {\partial y_2} {\partial W_{11}} & \frac {\partial y_2} {\partial W_{21}} &\cdots&\frac {\partial y_2} {\partial W_{nm}} \\
\cdots& \cdots &\cdots&\cdots \\
\frac {\partial y_m} {\partial W_{11}} & \frac {\partial y_m} {\partial W_{21}} &\cdots&\frac {\partial y_m} {\partial W_{nm}}
\end{bmatrix} 
=\begin{bmatrix}
x_1 & x_2 & \cdots & x_n&0&0&\cdots&0 \\
0& 0 & \cdots &0&x_1&x_2&\cdots&0 \\
\cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots &\\
0& 0 & \cdots &0&0&0&\cdots&x_n \\
\end{bmatrix} \\
\text{let } \frac {\partial L} {\partial y} =\begin{bmatrix}
\frac {\partial L} {\partial y_1},&\cdots,&\frac {\partial L} {\partial y_m}  
\end{bmatrix} \\
$$

$$
\text{then we have: }
\frac {\partial L} {\partial W} = \frac {\partial L} {\partial y} \frac {\partial y} {\partial W} = \begin{bmatrix} 
x_1\frac {\partial L} {\partial y_1} &x_2\frac {\partial L} {\partial y_1}&\cdots&x_n\frac {\partial L} {\partial y_1} \\
x_1\frac {\partial L} {\partial y_2} &x_2\frac {\partial L} {\partial y_2}&\cdots&x_n\frac {\partial L} {\partial y_2} \\
\cdots&\cdots&\cdots&\cdots&\\
x_1\frac {\partial L} {\partial y_m} &x_2\frac {\partial L} {\partial y_m}&\cdots&x_n\frac {\partial L} {\partial y_m} \\
\end{bmatrix} \text{(reshaped from [1,mn] to [m,n])}\\
\text{we found it equals }\frac {\partial L} {\partial W} =x'\frac {\partial L}{\partial y}
$$

$$
\text{similarly, }
\frac {\partial L} {\partial b} = \frac {\partial L} {\partial y} \frac {\partial y} {\partial b} = \frac {\partial L} {\partial y} 1= \frac {\partial L} {\partial y} \\
\frac {\partial L} {\partial x} = \frac {\partial L} {\partial y} \frac {\partial y} {\partial x} = \frac {\partial L} {\partial y}W'
$$

```python
def forward(x, W, b):
    """
    Input:
    	x: [1, n]
    	W: [n, m]
    	b: [1, m]
    Output:
    	y: [1, m]
    """
    y = x @ W + b
    return y

def backward(dy, x, W, b):
    """
    Input:
    	dy: [1, m]
    Output:
    	dx: [1, n]
    	dW: [n, m]
    	db: [1, m]
    """
    dx = dy @ W.T
    dW = x.T @ dy
    db = dy
    return dx
```

##### Batch

$X:[B, n],\ Y:[B, m],\ W: [n,m],\ b:[1,m]$ （数据为行向量）

```python
def forward(X, W, b):
    """
    Input:
    	X: [B, n]
    	W: [n, m]
    	b: [1, m]
    Output:
    	Y: [B, m]
    """
    Y = X @ W + b # broadcast add
    return Y

def backward(dY, X, W, b):
    """
    Input:
    	dY: [B, m]
    Output:
    	dX: [B, n]
    	dW: [n, m]
    	db: [1, m]
    """
    dX = dY @ W.T
    dW = X.T @ dY
    db = dY.sum(0)
    return dX
```



### Softmax & Cross entropy





### Convolution

