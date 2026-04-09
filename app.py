import streamlit as st
import cv2
import numpy as np
from PIL import Image
import kociemba

# 页面配置
st.set_page_config(page_title="魔方视觉还原助手", layout="centered")

# --- 1. 颜色与魔方状态定义 ---
# 采用标准的西方魔方配色：上白(U), 右红(R), 前绿(F), 下黄(D), 左橙(L), 后蓝(B)
COLOR_TO_FACE = {
    'white': 'U', 'red': 'R', 'green': 'F',
    'yellow': 'D', 'orange': 'L', 'blue': 'B', 'unknown': '?'
}

def detect_color_hsv(h, s, v):
    """根据 HSV 值判断颜色，这里的阈值需要根据实际光照微调"""
    if s < 50 and v > 120: return 'white'
    if s < 50 and v <= 120: return 'unknown' # 避免识别出黑色或灰色
    
    if (0 <= h <= 10) or (160 <= h <= 180): return 'red'
    elif 11 <= h <= 25: return 'orange'
    elif 26 <= h <= 35: return 'yellow'
    elif 36 <= h <= 89: return 'green'
    elif 90 <= h <= 130: return 'blue'
    return 'unknown'

# --- 2. 图像处理与色块提取 ---
def extract_cube_colors(image_pil):
    """从拍摄的照片中心提取 3x3 的色块矩阵"""
    # 将 PIL 图像转为 OpenCV 格式 (RGB -> BGR)
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    height, width, _ = img_cv.shape
    # 假设用户将魔方放在画面正中央，取中心正方形区域
    size = min(height, width) // 2
    start_x = width // 2 - size // 2
    start_y = height // 2 - size // 2
    
    step = size // 3
    face_colors = []
    
    # 遍历 3x3 的网格
    for row in range(3):
        for col in range(3):
            # 计算每个小方块的中心点
            cx = start_x + col * step + step // 2
            cy = start_y + row * step + step // 2
            
            # 取中心 5x5 区域的均值
            roi = hsv_img[cy-2:cy+3, cx-2:cx+3]
            avg_hsv = np.mean(roi, axis=(0, 1))
            
            color_name = detect_color_hsv(avg_hsv[0], avg_hsv[1], avg_hsv[2])
            face_colors.append(COLOR_TO_FACE[color_name])
            
            # 在原图上画框标记采样的点，用于调试展示
            cv2.rectangle(img_cv, (cx-10, cy-10), (cx+10, cy+10), (255,255,255), 2)
            cv2.putText(img_cv, COLOR_TO_FACE[color_name], (cx-10, cy-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            
    img_debug = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return face_colors, img_debug

# --- 3. 状态管理初始化 ---
if 'cube_state' not in st.session_state:
    # Kociemba 算法要求的面顺序: U, R, F, D, L, B
    st.session_state.cube_state = {'U': [], 'R': [], 'F': [], 'D': [], 'L': [], 'B': []}

# --- 4. UI 界面 ---
st.title("🧩 魔方视觉还原助手")
st.markdown("""
**操作说明：**
1. 请在光线充足的环境下进行。
2. 按照 **上(U)、右(R)、前(F)、下(D)、左(L)、后(B)** 的顺序拍摄魔方。
3. 拍摄时，请将魔方尽量填满画面中心的虚拟九宫格区域。
""")

# 定义面的顺序和提示语
faces_order = ['U', 'R', 'F', 'D', 'L', 'B']
face_names = {'U': '顶面 (白底)', 'R': '右面 (红底)', 'F': '前面 (绿底)', 
              'D': '底面 (黄底)', 'L': '左面 (橙底)', 'B': '后面 (蓝底)'}

# 让用户选择当前正在录入哪一个面
current_face = st.selectbox("当前录入面：", faces_order, format_func=lambda x: face_names[x])

camera_photo = st.camera_input(f"拍摄 {face_names[current_face]}")

if camera_photo is not None:
    img = Image.open(camera_photo)
    colors, debug_img = extract_cube_colors(img)
    
    st.image(debug_img, caption="识别结果 (查看采样点是否准确映射了色块)")
    
    if '?' in colors:
        st.error("存在无法识别的颜色，请调整光线或魔方位置后重新拍摄！")
    else:
        st.success(f"{face_names[current_face]} 识别成功: {colors}")
        if st.button(f"保存 {face_names[current_face]} 状态"):
            st.session_state.cube_state[current_face] = colors
            st.rerun()

# --- 5. 状态展示与求解 ---
st.divider()
st.subheader("📋 当前魔方状态")

col1, col2, col3, col4, col5, col6 = st.columns(6)
columns = [col1, col2, col3, col4, col5, col6]

all_faces_recorded = True
for idx, face in enumerate(faces_order):
    with columns[idx]:
        st.write(f"**{face} 面**")
        if st.session_state.cube_state[face]:
            st.write(st.session_state.cube_state[face])
        else:
            st.write("未录入")
            all_faces_recorded = False

if all_faces_recorded:
    st.success("✅ 六个面已全部录入！")
    if st.button("🚀 开始计算还原步骤", type="primary"):
        try:
            # 将字典中的状态拼接成 Kociemba 算法需要的 54 字符长字符串
            cube_string = ""
            for face in faces_order:
                cube_string += "".join(st.session_state.cube_state[face])
            
            st.info(f"算法识别字符串: `{cube_string}`")
            
            # 调用算法求解
            solution = kociemba.solve(cube_string)
            
            st.balloons()
            st.subheader("🎯 还原步骤：")
            st.code(solution, language="text")
            st.markdown("""
            **符号说明：**
            * **U/D/L/R/F/B**: 顺时针旋转对应面 90 度
            * **U'/D'/...**: 逆时针旋转 90 度
            * **U2/D2/...**: 旋转 180 度
            """)
            
        except ValueError as e:
            st.error("❌ 魔方状态非法！可能是识别错误或魔方本身被拆开乱装过。请检查识别记录或重置。")
            st.write(f"错误详情：{e}")

if st.button("🗑️ 重置所有状态"):
    st.session_state.cube_state = {'U': [], 'R': [], 'F': [], 'D': [], 'L': [], 'B': []}
    st.rerun()
