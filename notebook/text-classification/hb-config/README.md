
<p align="center">
  <img src="images/logo.png" width=250>
</p>

# hb-config: easy to configure your python packge


hb-config is utility for easy to configure your python project especially **Deep Learning experiments**.  

## Feature

- Do not use any **boilerplate code**
- **Singleton** Class (immediately use anywhere)
- Supports formats: **.json and .yaml**
- Access property using **\_\_getattr\_\_** function ```Config.TOKEN```
- Edit property using **\_\_setattr\_\_** function ```Config.TOKEN = "{token}"```
- every config data's type is **dict**


## Install

using pip

```
$ pip install hb-config
```

or clone repository

```
git clone https://github.com/mike-sino/hb-config.git
python setup.py install
```

## Usage

- base path : config/
- config3.yml example

```yml
project: hb-config
example: true
bot:
  in_bot:
    test: haha
    simple: wow
```

- Using like dict
	- only one difference : Config["project"] -> Config.project
	- using get Config.get("project"), Config.get("project", {})
	- using set Config.project = "set value" 
- Add description with python annotation. `#` 

### Load config

```python
>>> from hbconfig import Config
>>> Config("config3")
>>> Config
Read config file name: config3.yml
{
    "project": "hb-config",
    "example": true,
    "bot": {
        "in_bot": {
            "test": "haha",
            "simple": "wow"
        }
    }
}
```

### Get
```
>>> Config.bot.in_bot
{
    "test": "haha"
    "simple": "wow"
}

>>> Config.project
'hb-config'

>>> Config.bot.in_bot.get("simple")
'wow'

>>> Config.bot.in_bot.get("not_exist_key", "default_value")
'default_value'
```

### Set 

- The config file does not change.

```
>>> Config.bot.in_bot
{
    "test": "haha"
    "simple": "wow"
}

>>> Config.bot.in_bot = "hello"
>>> Config.bot.in_bot
'hello'

```

### Description

- example config

```yml
project: hb-config    # project name
example: true         # is it example?
bot:
  in_bot:
    test: haha
    simple: wow
```

```
>>> Config
Read config file name: ./config/config
{
    "project": "hb-config",
    "example": true,
    "bot": {
        "in_bot": {
            "test": "haha",
            "simple": "wow"
        }
    }
}
>>> Config.description
{'example': 'is it example?', 'project': 'project name'}
```


