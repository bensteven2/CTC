
def get_file_path(path_root):
    """获得svs文件的路径以及对应的xml文件"""

    all_file_path = []
    for item in path_root.iterdir():

        Filepaths_In_item = []  # 存放子文件夹下的所有xml文件和svs文件

        xmlpaths_In_item = list(item.glob('**/*.XML'))
        svs_path = list(item.glob('**/*.svs'))

        Filepaths_In_item = xmlpaths_In_item + svs_path

        if len(Filepaths_In_item) <= 1:  # 除去没有xml文件的文件夹
            continue
        Filepaths_In_item = [str(path) for path in Filepaths_In_item]  # windowsPath====>string类型

        all_file_path.append(Filepaths_In_item)

    print("Func:Get_File_Path() getting...")
    return all_file_path


if __name__ == "__main__":

    get_file_path(path_root)
