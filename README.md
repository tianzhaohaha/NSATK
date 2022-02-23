NSATK
###GCN模型
这里GCN模型我使用的是两层模型，为了便于修改，我使用的是原论文的GCN模型。
GCN模型更新公式：
$$H^{(l+1)}=\sigma(\tilde{D^{-1/2}}A\tilde{D^{-1/2}}H^{(l)}W^{(l)})$$

其中：
$\tilde{D }=D + I $
$\tilde{A }=A + I $
H为变化的特征
$A $是做了无向处理的邻接矩阵
$D_i = \sum_jA_{ij}$
>我这里采用了D的简单归一化，即$\tilde{D^{-1}}A$替代$\tilde{D^{-1/2}}A\tilde{D^{-1/2}}$

一层hidden layer的GCN公式：   

$$Z=f_{\theta}(A,X)=softmax(\hat{A}\sigma(\hat{A}XW^{(1)})W^{(2)})$$

###ATTACK MODEL
####1.GMA on node set(graph modify attack)
首先指定一组target nodes，对于direct attack受攻击节点就是target nodes：$V$，对于influential attack则是target nodes的邻居节点作为：$V$

对于GMA形式的攻击：
$$A^{'}_{vu}\ne A^0_{vu}  \ \ where\ \ u\in V \ \ and \ \ v\not \in V  $$
对于上式：v代表受攻击节点V集合内的节点，u代表与V连接的图上其他节点，$A^{'}$代表经过扰动的图，$A^0$代表原图，值得一提的是，我在训练开始将图转为了无向图，所以$A_{vu} = A_{uv}$



接下来，对于攻击后的图，使用训练好的GCN权重W1，W2进行预测（在整个攻击过程权重不变），目标是：
$$argmax \sum_{v\in V}（ Z_{v,c}- Z_{v,c}^*）$$
$$Z = f_{\theta}(A,X)\ \ and \ \ Z* = f_{\theta}(A^{'},X^{'}) $$
这里，$Z_{v,c}$代表GCN模型，对于节点v在标签c上的预测值，attack model的目的是最大化上述括号内的公式。


####2.弱化扰动联系
**概述**：在扰动后的图训练一个改进后的deepwalk模型，按某种方式剔除一些扰动，再使用deepwalk预测余下的出现概率。变动较大的扰动剔除。

这里我使用的是以skip-gram为内核的deepwalk算法，deepwalk算法并不能直接用于弱化扰动联系，其存在一些问题：
（1）deepwalk算法不牵扯节点feature计算
（2）deepwalk算法是求节点embedding算法，不能直接使用

**skip-gram模型**：对于此模型，其词汇（节点）相似度公式为：

$$softmax(x_k \times W_{V\times N}\times W_{N\times V}^{'})$$
此公式求的是词汇（节点）之间相似度，$x_k$词汇（节点）one-hot编码；$ W_{V\times N}$代表所有词汇的词汇编码的矩阵；$W_{N\times V}^{'}$代表前者的转置。此公式结果是词汇之间的pearson相似度的值。也代表可能出现的概率。

对于任意一个节点$u_k$，出现于节点$u_c$概率是：
$$P(u_k|u_c) = \frac{exp(W^{'}_k\times W_c)}{\sum_{c=1}^V exp(W_c\times W^{'}_k)}$$
这里$W^{'}_k$代表转置矩阵第k列；$W_c$代表原词汇矩阵第c行。

我所做的，就是使用deepwalk将图节点embedding训练好之后，除去部分扰动，使用skipgram推测出剩余扰动出现概率，并据此决定留存。









