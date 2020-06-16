import xml.dom.minidom


def load_xml(file_path):
    """读取xml文件，
    返回总坐标列表xy_list:存放一张图片上所有画出得区域的点 """

    # 用于打开一个xml文件，并将这个文件对象dom变量
    dom = xml.dom.minidom.parse(file_path)
    # 对于知道元素名字的子元素，可以使用getElementsByTagName方法获取
    Annotations = dom.getElementsByTagName('Annotation')

     ## 存放所有的 Annotation

    XYI_in_Annotations = []
    XYN_in_Annotations = []

    for Annotation in Annotations:
        #print("      Load {0}th area, Area_Name:{1}".format(i_area,Annotation.getAttribute("Name")))


        # 存放一个 Annotation 中所有的 X,Y值
        XY_in_Annotation = []

        # 读取 Coordinates 下的 X Y 的值
        Coordinates = Annotation.getElementsByTagName("Coordinate")
        for Coordinate in Coordinates:
            list_in_Annotation = []
            X=int(float(Coordinate.getAttribute("X")))
            Y=int(float(Coordinate.getAttribute("Y")))

            list_in_Annotation.append(X)
            list_in_Annotation.append(Y)


            XY_in_Annotation.append(list_in_Annotation)


        Name_Area=Annotation.getAttribute("Name")
        if Name_Area=="normal":
            XYN_in_Annotations.append(XY_in_Annotation)
        if Name_Area != "normal":
            XYI_in_Annotations .append(XY_in_Annotation)

        XY_tuple= (XYI_in_Annotations,XYN_in_Annotations)
        # xy_list.append(XYI_in_Annotations)
        # xy_list.append(XYN_in_Annotations)

    return XY_tuple


if __name__ == "__main__":

    load_xml(all_file_path)