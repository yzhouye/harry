import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px

# 读取 JSON 文件
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 家族关系可视化
# 生成关系图
def create_relationship_graph(data):
    G = nx.DiGraph()
    
    # 遍历每个人物，添加其家庭信息中的关系
    for name, details in data.items():
        G.add_node(name)  # 添加人物节点
        
        if "家庭信息" in details:
            family_info = details["家庭信息"]
            for relationship, connected_people in family_info.items():
                if isinstance(connected_people, list):
                    for cp in connected_people:
                        G.add_edge(name, cp, relation=relationship)
                else:
                    G.add_edge(name, connected_people, relation=relationship)
    return G

# 递归查找与起始人物相关的子图
def get_subgraph(G, start_person):
    subgraph = nx.DiGraph()
    visited = set()
    
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        subgraph.add_node(node)
        
        # 递归遍历相邻节点
        for neighbor in G.neighbors(node):
            relation = G[node][neighbor]["relation"]
            subgraph.add_node(neighbor)
            subgraph.add_edge(node, neighbor, relation=relation)
            dfs(neighbor)
    
    dfs(start_person)
    return subgraph

# 计算网络属性
def calculate_properties(G):
    properties = {}

    # 1. 平均度
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    properties["平均度"] = avg_degree

    # 2. 平均加权度
    # 假设每条边的权重为 1（如果是加权图，权重需要从边属性中读取）
    avg_weighted_degree = sum(d for n, d in G.degree(weight="weight")) / G.number_of_nodes()
    properties["平均加权度"] = avg_weighted_degree

    # 3. 密度
    density = nx.density(G)
    properties["密度"] = density

    # 4. 平均聚类系数
    avg_clustering_coefficient = nx.average_clustering(G)
    properties["平均聚类系数"] = avg_clustering_coefficient

    # 5. 直径
    try:
        diameter = nx.diameter(G.to_undirected())  # 有向图转换为无向图计算直径
    except:
        diameter = float("inf")  # 如果图不连通，直径为无穷大
    properties["直径"] = diameter

    # 6. 平均路径长度
    try:
        avg_shortest_path_length = nx.average_shortest_path_length(G.to_undirected())  # 有向图转换为无向图
    except:
        avg_shortest_path_length = float("inf")
    properties["平均路径长度"] = avg_shortest_path_length

    # 7. 特征向量中心度
    try:
        eigenvector_centralities = nx.eigenvector_centrality(G)
        for node, centrality in eigenvector_centralities.items():
            properties["特征向量中心度"] = centrality
            break
    except:
        pass

    return properties

# 可视化子图
def visualize_subgraph(G, start_person):
    plt.figure(figsize=(18, 12))
    pos = nx.kamada_kawai_layout(G, scale=10)

    # 绘制节点
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="#5F9BED", font_size=18, font_weight="bold",
            width=1.0, alpha=0.8, font_color="#495057", edge_color="#9198A1", font_family="Microsoft YaHei")
    
    # 绘制边和边标签
    edge_labels = {(source, target): G[source][target]["relation"] for source, target in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='purple', font_size=12, font_family="Microsoft YaHei")
    
    # 显示图形
    st.pyplot(plt.gcf())

# 属性关系可视化
# 提取人物的属性并转换为 DataFrame
def extract_character_attributes(data):
    attributes = []
    for name, details in data.items():
        attributes.append({
            "姓名": name,
            "性别": details.get("性别", "无"),
            "血统": details.get("血统", "未知"),
            "物种": details.get("物种", "未知"),
            "学院": details.get("学院", "未知")
        })
    return pd.DataFrame(attributes)

def visualize_attributes(data):
    # 提取人物属性
    df = extract_character_attributes(data)

    # 属性列表
    attributes = ["性别", "血统", "物种", "学院"]

    # 遍历每个属性
    for attribute in attributes:
        st.markdown(f"- #### {attribute} 属性")

        # 统计选中属性的分布
        stats = df[attribute].value_counts()

        # 排除“未知”值
        if attribute == "血统" or attribute == "物种" or attribute == "学院":
            stats = stats[stats.index != "未知"]
        if attribute == "物种":
            stats = stats[stats.index != "人类"]

        # 绘制饼图
        fig_pie = px.pie(
            stats,
            names=stats.index,
            values=stats.values,
            title=f"{attribute} 分布（饼图）",
            color_discrete_sequence=px.colors.qualitative.Plotly,
        ).update_traces(
            hovertemplate=
            "<b>%{label}</b><br>" +  # 显示类别名称
            "数量: %{value}<br>" +  # 显示数量
            "百分比: %{percent:.2%}"  # 显示百分比
        )

        # 绘制柱状图
        fig_bar = px.bar(
            stats,
            x=stats.index,
            y=stats.values,
            title=f"{attribute} 分布（柱状图）",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            color=stats.index,  # 按类别着色
            text=stats.values,  # 显示柱子上的数值
            text_auto=True,  # 自动显示数值
        ).update_traces(
            hovertemplate=  # 鼠标悬停时的模板
            "<b>%{x}</b><br>" +  # 显示类别名称
            "数量: %{y}<br>"  # 显示数量
        ).update_layout(
            xaxis_title=attribute,  # 设置 x 轴标题
            yaxis_title="数量",  # 设置 y 轴标题
        )

        # 并排显示饼图和漏斗图
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(fig_pie)
        with col2:
            st.plotly_chart(fig_bar)

# 社交关系可视化
# 筛选含有职业信息的人物
def extract_person_with_occupation(data):
    persons_with_occupation = {}
    for name, info in data.items():
        if "职业" in info and info["职业"]:
            persons_with_occupation[name] = info["职业"]
    return persons_with_occupation

# 筛选含有从属信息的人物
def extract_person_with_deparment(data):
    persons_with_deparment = {}
    for name, info in data.items():
        if "从属" in info and info["从属"]:
            persons_with_deparment[name] = info["从属"]
    return persons_with_deparment

# 计算重合度
def calculate_overlap(persons_with_occupation):
    overlap_matrix = defaultdict(dict)
    names = list(persons_with_occupation.keys())
    
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            person1 = names[i]
            person2 = names[j]
            overlap = len(set(persons_with_occupation[person1]) & set(persons_with_occupation[person2]))
            if overlap > 0:
                overlap_matrix[person1][person2] = overlap
                overlap_matrix[person2][person1] = overlap
    return overlap_matrix

# 使用 NetworkX 构建图
def create_network(overlap_matrix):
    G = nx.Graph()
    for person1, connections in overlap_matrix.items():
        for person2, weight in connections.items():
            G.add_edge(person1, person2, weight=weight)
    return G

# 使用 Plotly 可视化社交网络图
def visualize_network(G):
    # 使用 Fruchterman-Reingold 布局生成节点位置
    pos = nx.fruchterman_reingold_layout(G, k=1.5)
    
    # 节点数据
    node_x = []
    node_y = []
    for node in G.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    # 节点标记
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=15,
            color='rgba(102,165,221, 0.8)',
            line=dict(width=1, color='rgb(50,50,50)')
        ),
        text=list(G.nodes),  # 显示节点名称
        hoverinfo='text',
        textposition="middle center"
    )
    
    # 边数据
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color='#eee'),
        hoverinfo='none',
        mode='lines'
    )
    
    # 配置布局
    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    # 组合图
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    st.plotly_chart(fig)

# 目录
def show_directory():
    # 在侧边栏展示目录
    st.sidebar.markdown("# 目录", unsafe_allow_html=True)  # 一级标题
    st.sidebar.markdown("---")  # 分隔线

    st.sidebar.markdown("## 1. 家族关系可视化")  # 二级标题
    #st.sidebar.markdown("- **3D 图谱展示**")
    st.sidebar.markdown("- **家族关系图谱**")
    st.sidebar.markdown("- **人物关系详述**")

    st.sidebar.markdown("## 2. 属性关系可视化")  # 二级标题
    st.sidebar.markdown("- **性别属性**")
    st.sidebar.markdown("- **血统属性**")
    st.sidebar.markdown("- **物种属性**")
    st.sidebar.markdown("- **学院属性**")

    st.sidebar.markdown("## 3. 社交关系可视化")  # 二级标题
    st.sidebar.markdown("- **职业社交圈**")
    st.sidebar.markdown("- **从属社交圈**")

    page_bg_img = '''
        <style>
        body {
        background-image: url("media/background.jpg");
        background-size: cover;
        }
        </style>
        '''
        
    st.markdown(page_bg_img, unsafe_allow_html=True)

# 主角介绍
def show_characters():
    p1 = "media/哈利.jpg"
    p2 = "media/哈利1.jpg"
    p3 = "media/赫敏.jpg"
    p4 = "media/赫敏1.jpg"
    p5 = "media/罗恩.jpg"
    p6 = "media/罗恩1.jpg"
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(p1)
    with col2:
        st.markdown("")
        st.markdown("*哈利·波特 (Harry Potter)，全名为哈利·詹姆·波特 (Harry James Potter)，生于1980年7月31日，是一个混血统巫师，也是现代最有名的巫师之一。他是凤凰社成员詹姆和莉莉·波特唯一的孩子。*")
        st.image(p2)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("*赫敏·格兰杰 (Hermione Granger) (生于1979年9月19日)，全名为赫敏·珍·格兰杰 (Hermione Jean Granger)，是麻瓜格兰杰夫妇的独生女，他们一家人居住在伦敦。十一岁时，赫敏得知自己是一个女巫，需要去霍格沃茨魔法学校上学。她在1991年9月1日进入学校读书，被分进格兰芬多学院。她精通学术、用功好学，被认为是天才，但是同时又有些书呆子气。和哈利·波特还有其他一些人一样，赫敏很少会被伏地魔的名字吓倒，敢于直呼其名，而不是使用“神秘人”或者“那个连名字都不能提的魔头”。*")
        st.image(p4)
    with col2:
        st.markdown("")
        st.image(p3)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(p5)
    with col2:
        st.markdown("")
        st.markdown("*罗恩·韦斯莱 (Ron Weasley)，全名为罗纳德·比利尔斯·韦斯莱 (Ronald Bilius Weasley)，出生于1980年3月1日，是一个纯血统巫师。罗恩在1991年进入霍格沃茨魔法学校学习，被分进格兰芬多学院。他很快就先后与哈利·波特和赫敏·格兰杰成为了好朋友。*")
        st.image(p6)

# 主程序
def main():
    # 设置全局字体
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.facecolor'] = '#f0f0f0'

    show_directory()
    st.markdown("# 哈利·波特人物关系可视化")
    st.audio("media/哈利·波特与魔法石.mp3", autoplay=True)
    st.markdown("---")
    show_characters()

    # 家族关系可视化
    st.markdown("---")
    st.markdown("## 1. 家族关系可视化")
    #st.markdown("- ### **3D 图谱展示**")
    #video_file_path = "media/3D图谱视频.mp4"
    #st.video(video_file_path)
    st.markdown("- ### **家族关系图谱**")
    st.markdown("> 选择一个***起始人物***，展示与其相关的家庭关系图。")
    
    # 加载数据
    file_path = "data/Harry.json"
    data = load_data(file_path)
    
    # 构建完整的关系图
    G = create_relationship_graph(data)
    
    # 用户输入起始人物
    all_names = [name for name, details in data.items() if "家庭信息" in details]
    start_person = st.selectbox("请选择一个起始人物",all_names)
    
    # 检查起始人物是否有效
    if start_person not in G:
        st.warning("起始人物不在数据中，请重新选择。")
        return
    
    # 生成与起始人物相关的子图
    subgraph = get_subgraph(G, start_person)
    
    # 显示子图
    st.write(f"> 与 ***{start_person}*** 相关的家庭关系图")
    visualize_subgraph(subgraph, start_person)
    properties = calculate_properties(subgraph)
    for key, value in properties.items():
        st.write(f"**{key}**: {value}")
    
    # 显示子图的详细信息
    st.markdown("- ### **人物关系详述**")
    st.write(f"> 子图含有 ***{subgraph.number_of_nodes()}*** 个节点， ***{subgraph.number_of_edges()}*** 条边。")

    edge_details = []
    for source, target in subgraph.edges():
        relation = subgraph[source][target]["relation"]
        edge_details.append({
            "源节点": source,
            "关系": relation,
            "目标节点": target
        })

    # 将数据转换为 DataFrame
    df_edge_details = pd.DataFrame(edge_details)

    # 检查是否有数据
    if df_edge_details.empty:
        st.write("暂无家庭关系数据。")
    else:
        st.write("> **边的详细信息:**")
        st.dataframe(df_edge_details)

    # 属性关系可视化
    st.markdown("---")
    st.markdown("## 2. 属性关系可视化")
    st.markdown("> **人物数据预览:**")
    df = extract_character_attributes(data)
    st.dataframe(df)
    visualize_attributes(data)
    
    # 社交关系可视化
    st.markdown("---")
    st.markdown("## 3. 社交关系可视化")
    st.markdown("- ### **职业社交圈**")
    # 筛选含有职业信息的人物
    persons_with_occupation = extract_person_with_occupation(data)
    # 计算职业重合度
    overlap_matrix = calculate_overlap(persons_with_occupation)
    # 生成社交网络图
    G1 = create_network(overlap_matrix)
    # 可视化社交网络图
    visualize_network(G1)
    st.markdown("- ### **从属社交圈**")
    # 筛选含有从属信息的人物
    persons_with_department = extract_person_with_deparment(data)
    # 计算职业重合度
    overlap_matrix = calculate_overlap(persons_with_department)
    # 生成社交网络图
    G2 = create_network(overlap_matrix)
    # 可视化社交网络图
    visualize_network(G2)


# 运行程序
if __name__ == "__main__":
    main()
