# 细粒度情感分析

# pipline


# joint
    absa_joint_parallel_bertcrf：两个标注任务并行
        优点：一个模型
        缺点：两个标注任务难易程度不一，合并loss权重不易调节
        
    absa_joint_series_bertcrf：两个标注任务串行
        优点：两个标注任务分别训练，互相不会干扰
        缺点：两个模型，较大；解决办法可以采用bert前11层不训练且共享
    
# unified
    absa_bertbigrucrf
    absa_bertcrf
    absa_bertgru
    absa_bertlinear
    论文：
        Exploiting BERT for End-to-End Aspect-based Sentiment Analysis
        A Unified Model for Opinion Target Extraction and Target Sentiment Predictio
        
# mrc: 基于阅读理解任务的方法，此方法来源于嵌套命名实体识别最新成果
    “find the term" + sentence --> 标注任务为寻找实体项目
    "tag the sentiment" + sentence --> 标注任务为标记情感