使用模板匹配任务的前提：找到和自己任务中相匹配的模板

# 使用到的技术：
1. 轮廓检测：外轮廓 + 外接矩形 
2. 模板匹配：resize(外接矩形 和 模板 resize成相同大小)

# 项目流程
1. 准备模板（根据任务的不同选择不同的模板进行匹配）
2. 对模板进行轮廓检测，选择外轮廓，得到外轮廓的外接矩形。
3. 对需要识别的卡片图片也做外轮廓检测，同样得到其外接矩形。
4. 将模板中的外接矩形和卡片中的外接矩形进行 resize 操作，使其变成相同大小。
5. 卡片中的外接矩形和模板中的10个外接矩形依次比较，相似程度大的就说明匹配成功。

# 代码

https://www.yuque.com/darrenzhang/cv/dnxxcq