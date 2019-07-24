from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileRequired


# TODO Refactor the form
class RequestForm(FlaskForm):
    name = StringField('What is your name?', validators=[DataRequired()])
    video = FileField(validators=[FileRequired()])
    submit = SubmitField('Submit')