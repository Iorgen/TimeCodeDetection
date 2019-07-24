import os
from flask_bootstrap import Bootstrap
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileRequired
from core.controller import Controller

# TODO figue out to app inizialization - For CPU support only
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

bootstrap = Bootstrap(app)
controller = Controller()


# TODO Refactor the form
class RequestForm(FlaskForm):
    name = StringField('What is your name?', validators=[DataRequired()])
    video = FileField(validators=[FileRequired()])
    submit = SubmitField('Submit')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)


@app.route('/', methods=['POST', 'GET'])
def recognition():
    name = None
    form = RequestForm()
    if form.validate_on_submit():
        name = form.name.data
        video = form.video.data
        predictions = controller.video_recognition(video)
        print(predictions)
        return render_template("results.html", predictions=predictions,  init=True)
    return render_template('index.html', form=form)



if __name__ == '__main__':
    app.run(debug=True)
