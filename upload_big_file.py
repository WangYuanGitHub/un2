import os, time
import commentjson as json
from onedrive_util import OneDrive  #  pip install Office365-REST-Python-Client
from datetime import datetime
import shutil

# https://pypi.org/project/onedrive-sharepoint/
def get_onedrive_handle():
    # 第一步获取OneDrive访问句柄
    email = "*******@yiteam.tech"
    password = input('Enter administrator password please?')
    endpoint = "https://ageasga-my.sharepoint.com/personal/*******_yiteam_tech"
    type = "onedrive"
    session = OneDrive(email=email, password=password, endpoint=endpoint, type=type)
    return session

session = get_onedrive_handle()

def add_file_to_onedrive(session, key, path_file_name_local):
    for retry in range(5):
        try:
            # 上传文件
            shareserver_remote_root = 'ShareServer/'
            print(f'uploading local file {path_file_name_local} with remote key {key}')
            session.upload_file_on_folder(path_file_name_local, shareserver_remote_root)

            # 创建公开链接
            share_link = session.share_folder(os.path.join(shareserver_remote_root, os.path.basename(path_file_name_local)), is_edit=False)

            # 读取manifest目录
            dir_name = f"./{ datetime.now().strftime('%d-%m-%Y') }-datas"
            session.download_file('/personal/*******_yiteam_tech/Documents/ShareServer/uhmap_manifest.jsonc')
            manifest_path_local = os.path.relpath( os.path.join(dir_name, 'uhmap_manifest.jsonc') ).replace('\\','/')
            with open(manifest_path_local, 'r', encoding='utf8') as f:
                manifest = json.load(f)
            manifest[key] = share_link

            with open(manifest_path_local, 'w', encoding='utf8') as f:
                json.dump(manifest, f, indent=4)

            # 上传manifest目录
            session.upload_file_on_folder(manifest_path_local, shareserver_remote_root)

            print('success')
            break
        except:
            print("upload fail")

def get_current_version():
    with open('current_version', 'r', encoding='utf8') as f:
        version = f.read()
        return version

desired_version = get_current_version()

def shutil_rmtree(p):
    if os.path.exists(p): 
        shutil.rmtree(p)

"""
检查：是否注册了地图
检查：是否注册了智能体
"""



plat = "Linux"
key = f"Uhmap_{plat}_Build_Version{desired_version}"
shutil_rmtree('./Build/LinuxNoEditor')
shutil_rmtree('./Build/LinuxServer')
os.system('python BuildLinuxRender.py')
os.system('python BuildLinuxServer.py')
os.system(f'.\\7-Zip\\7z.exe a -tzip -mx4 ./Build/{key}.zip  ./Build/LinuxNoEditor   ./Build/LinuxServer')



plat = "Windows"
key = f"Uhmap_{plat}_Build_Version{desired_version}"
shutil_rmtree('./Build/WindowsNoEditor')
shutil_rmtree('./Build/WindowsServer')
os.system('python BuildWindowsRender.py')
os.system('python BuildWindowsServer.py')
os.system(f'.\\7-Zip\\7z.exe a -tzip -mx4 ./Build/{key}.zip  ./Build/WindowsNoEditor   ./Build/WindowsServer')

add_file_to_onedrive(
    session = session, 
    key = key, 
    path_file_name_local = f'Build/{key}.zip')

add_file_to_onedrive(
    session = session, 
    key = key, 
    path_file_name_local = f'Build/{key}.zip')

os.system(f'.\\7-Zip\\7z.exe a -tzip -mx4 ./Build/uhmp-big-file-v{desired_version}.zip  -ir!Content/Model3D   Plugins  7-Zip  UHMP.uproject  EnvDesignTutorial.pptx')
add_file_to_onedrive(
    session = session, 
    key = f'uhmp-big-file-v{desired_version}', 
    path_file_name_local = f'Build/uhmp-big-file-v{desired_version}.zip')