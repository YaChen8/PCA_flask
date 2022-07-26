import os.path
from flask import request, Flask, jsonify
from flask import Flask,flash
from flask import request
from flask import render_template
# import Pca_feature_recognition
import mycode.pca_feature
import pca_feature
from werkzeug.utils import secure_filename
app = Flask(__name__) #创建一个Web应用的实例”app”

@app.route('/')
def index():
    return '<h1>Hello World</h1>'

@app.route('/user/<int:user_id>')
def get_user(user_id):
    return 'User ID: %d' % user_id

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return 'This is a POST request'
    else:
        return 'This is a GET request'

@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

@app.route('/testnext')
def testnext():
    return render_template('next.html')

@app.route('/test')
def get_result():
    sort_weightValues, sort_filedires = pca_feature.myPCA("FaceDB_orl","FaceDB_orl/001/10.png")
    # print(sort_weightValues, sort_filedires)
    # 展示前三个最相似的
    print_str=""
    for i in range(3):
        print_str+=str(sort_weightValues[i])+"\t\t"+sort_filedires[i]
        print_str+='\n'
    # return str(sort_weightValues[0])+sort_filedires[0]
    # print(print_str)
    return print_str

basedir = os.path.abspath(os.path.dirname(__file__))
uploadDir = os.path.join(basedir, 'static/uploads')

@app.route('/testpca', methods=['POST','GET'])
def pca():
    if request.method == 'POST':
        f = request.files.get('fileupload')
        if not os.path.exists(uploadDir):
            os.makedirs(uploadDir)
        if f:
            filename = secure_filename(f.filename)
            types = ['jpg','png','tif']
            if filename.split('.')[-1] in types:
                uploadpath = os.path.join(uploadDir,filename)
                f.save(uploadpath)
                # flash('Upload Load Successful!','success')
                print('Upload Load Successful!')
                print(uploadpath)
                sort_weightValues, sort_filedires = pca_feature.myPCA("FaceDB_orl", uploadpath)
                print(sort_filedires[0],sort_filedires[1],sort_filedires[2])
                first = sort_filedires[0].split("\\")
                second = sort_filedires[1].split("\\")
                third = sort_filedires[2].split("\\")
                fourth = sort_filedires[3].split("\\")
                first1 = first[-2]
                first2 = first[-1]
                second1 = second[-2]
                second2 = second[-1]
                third1 = third[-2]
                third2 = third[-1]
                fourth1 = fourth[-2]
                fourth2 = fourth[-1]
            else:
                # flash('Unknow Types!','danger')
                print('Unknow Types!')
        else:
            # flash('No File Selected','danger')
            print('No File Selected')
        return render_template('next.html',imagename=filename,
                               first1=first1,first2=first2,
                               second1=second1,second2=second2,
                               third1=third1,third2=third2,
                               fourth1=fourth1,fourth2=fourth2)
    return render_template('index.html')

@app.route('/getdir',methods=['POST'])
def getTestDir():
    testPath = request.form['path']
    print(testPath)
    return testPath

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)