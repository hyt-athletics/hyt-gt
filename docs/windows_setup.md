# Windows 11 安装与使用指南

## 第一步：安装 Python

1. 打开浏览器，访问 https://www.python.org/downloads/
2. 点击 **Download Python 3.11.x**（或更高版本）
3. 运行下载的安装程序
4. **重要：勾选 ☑ Add Python to PATH**
5. 点击 Install Now，等待安装完成

验证安装成功：按 `Win+R`，输入 `cmd`，回车，然后输入：

```
python --version
```

看到 `Python 3.11.x` 就表示安装成功。

---

## 第二步：下载项目

### 方法 A：从 GitHub 下载 ZIP（最简单）

1. 打开 https://github.com/hyt-athletics/hyt-gt
2. 点击绿色的 **Code** 按钮
3. 点击 **Download ZIP**
4. 解压到你想放的位置，比如 `D:\geophys-tool\`

### 方法 B：使用 Git（推荐）

如果你安装了 Git（https://git-scm.com/download/win），在命令提示符中运行：

```
git clone https://github.com/hyt-athletics/hyt-gt.git
cd hyt-gt
```

---

## 第三步：安装依赖

打开命令提示符，进入项目目录：

```
cd D:\geophys-tool
pip install uv
uv sync
```

如需三维可视化功能（推荐）：

```
uv sync --extra 3d
```

如需神经网络算法支持：

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime
```

---

## 第四步：启动软件

### 方式 A：双击启动（推荐）

双击项目目录中的 `start.bat` 文件即可。

### 方式 B：命令行启动

```
cd D:\geophys-tool
python main.py
```

---

## 第五步：开始使用

1. 点击工具栏 **+ 新建钻孔**，输入名称（如 `BH-001`）
2. 点击 **导入数据**，选择你的 LAS / CSV 数据文件
3. 在右侧选择算法，设置参数，点击 **运行算法**
4. 结果自动显示在中央区域

详细操作请看：[用户使用手册](user_guide.md)

---

## 常见问题

### 启动时报错「找不到 PyQt6」

```
pip install PyQt6
```

### 启动时报错「找不到 numpy / scipy」

```
uv sync
```

或者手动安装：

```
pip install numpy scipy matplotlib pandas lasio segyio empymod pywavelets
```

### 界面显示不正常（太小或太大）

在 `start.bat` 前面加一行：

```
set QT_SCALE_FACTOR=1.25
```

数字可以调整：`1.0` = 100%，`1.25` = 125%，`1.5` = 150%

### 中文显示为方块

系统通常自带微软雅黑字体，重启软件即可。如果仍有问题：

- 打开 Windows 设置 → 时间和语言 → 语言和区域
- 确认已安装中文语言包

### 如何编译算法为动态库（.pyd）

需要先安装 Visual Studio Build Tools：

1. 打开 https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. 下载并安装，勾选「C++ 桌面开发」
3. 然后运行：

```
pip install cython
python tools\compile_algo.py algorithms\你的算法名
```

编译后会生成 `algorithm.cp311-win_amd64.pyd` 文件。

---

## 目录说明

```
geophys-tool\
├── start.bat              ← 双击启动软件
├── main.py                ← 软件入口
├── algorithms\            ← 算法模块（35个）
├── pipelines\             ← 流水线配置
├── tools\
│   ├── verify.bat         ← 双击验证算法包
│   ├── validate_algo.py   ← 算法验证工具
│   ├── scaffold_algo.py   ← 算法模板生成
│   └── compile_algo.py    ← 编译为动态库
├── docs\
│   ├── user_guide.md      ← 用户手册
│   ├── developer_guide.md ← 算法开发指南
│   └── full_report.md     ← 项目汇报文档
└── tests\                 ← 测试脚本
```

数据存放在用户目录下：

```
C:\Users\你的用户名\geophys-workspace\
└── boreholes\
    └── BH-001\
        ├── logging\raw\       ← 导入的数据
        └── logging\processed\ ← 算法输出
```
