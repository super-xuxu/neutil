from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
import scipy.io as sio
from matplotlib import pyplot as plt
from django.http import StreamingHttpResponse
from ne.ne_process import network_ne
from django.http import FileResponse
import os
import numpy as  np
import pandas as pd


# Create your views here.
def index(request):
    return render(request, "index.html")


def denoise_network(request):
    if request.method == "POST":  # 请求方法为POST时，进行处理

        myFile = request.FILES.get("file", None)  # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return HttpResponse("No files for upload!")

        address = save_file(myFile)

        # response
        data = {}

        data['address'] = address

        network = {}


        ext = address.split('.')[1]
        if ext == 'mat':
            network = sio.loadmat(address)

        elif  ext == 'txt':
            raw = np.loadtxt(address, dtype=np.float64, delimiter=",")
            network['raw']=raw

        elif ext == 'csv' :
            df=pd.read_csv(address)

            network['raw']=np.array(df.loc[:, :])
            network['label']=list(df.columns.values)





        data['image'] = address.split('.')[0] + '.png'
        data['ne_address'] = network_ne(address,network)
        data['ne_image']='ne_'+address.split('.')[0] + '.png'

        return JsonResponse(data)


'''
def file_download(request):
    # do something...
    for key in request.GET:
        print (key)
    the_file_name = request.GET.get("filename", None)
    print(the_file_name)

    def file_iterator(file_name, chunk_size=6000000):
        with open(file_name,'rb') as f:
            while True:
                c = f.read(chunk_size)
                if c:
                    yield c
                else:
                    break

    response = StreamingHttpResponse(file_iterator(the_file_name))
    response['Content-Length'] = str(os.path.getsize(the_file_name))
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format(the_file_name)

    return response
'''


def file_download(request):
    """
    Send a file through Django without loading the whole file into
    memory at once. The FileWrapper will turn the file object into an
    iterator for chunks of 8KB.
    """
    the_file_name = request.GET.get('filename')
    file = open(the_file_name, 'rb')

    response = FileResponse(file)
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Length'] = os.path.getsize(the_file_name)
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format(the_file_name)
    return response


def save_file(file):
    destination = open(file.name, 'wb+')
    for chunk in file.chunks():  # 分块写入文件
        destination.write(chunk)
    destination.close()
    address = file.name
    return address

