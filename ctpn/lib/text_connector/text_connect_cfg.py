class Config:
    SCALE = 900  # 600 # 图像大小
    MAX_SCALE = 1500  # 1200 # 图像最大尺寸
    TEXT_PROPOSALS_WIDTH = 0  # 16
    MIN_NUM_PROPOSALS = 0  # 2
    MIN_RATIO = 0.01  # 0.5
    LINE_MIN_SCORE = 0.6  # 0.9
    MAX_HORIZONTAL_GAP = 30  # 50 # 水平方向最大跨度
    TEXT_PROPOSALS_MIN_SCORE = 0.7  # 0.7 # 判定当前框为文字框的最低得分
    TEXT_PROPOSALS_NMS_THRESH = 0.3  # 0.2 # NMS非极大值抑制系数
    TEXT_LINE_NMS_THRESH = 0.3

    # 判断两个框是否在竖直方向重叠参数
    MIN_V_OVERLAPS = 0.6  # 0.7
    MIN_SIZE_SIM = 0.6  # 0.7
