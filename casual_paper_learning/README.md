# Casual Paper Learning Utils

## 如何使用
为了尽可能简化笔记文件夹，我把**需要用到的 python 文件**放到了该文件夹下进行管理，对于有的笔记(比如 DDPM)，会依赖这里的 python 包，因此在尝试运行复现之前，请安装此包:

```bash
cd casual_paper_learning
pip install -e .
```

## Visual Studio Code 可能遇到的问题
<u>**(可以自行选择是否解决，不影响代码的运行)**</u>

如果你使用 **Visual Studio Code**，可能在运行上述命令之后遇到类似的报错报错:

```text
无法导入 c_ddpm
```

但是你如果安装当前包之后，其实是可以正常运行的，只要你确保安装了此包:

```bash
pip show casual_paper_learning
```

若输出类似:

```text
Name: casual_paper_learning
Version: 0.1.0
...
```

就安装成功了

**解决方式如下**:

打开项目中的 .vscode/settings.json 文件，添加:

```json
"python.analysis.extraPaths": [
    "/PATH/TO/casual_paper_learning"
],
"python.autoComplete.extraPaths": [
    "/PATH/TO/casual_paper_learning"
],
```

替换上面的 `PATH` 为这个包的绝对路径，就能识别到包里的模块了

**如果还有其他问题，欢迎补充**
