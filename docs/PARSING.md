
- Log format for each dataset:
```yaml
'HDFS': {
  'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
  'regex': [ r'blk_-?\d+', r'(\d+\.){ 3 }\d+(:\d+)?' ]
}

'BGL': {
  'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
  'regex': [ r'core\.\d+' ]
}

'Thunderbird': {
  'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
  'regex': [ r'(\d+\.){ 3 }\d+' ]
}

'Spirit': {
  'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Content>',
  'regex': [ r'(\d+\.){ 3 }\d+', r'(\/.*?\.[ \S: ]+)' ],
}

```
- Configuration for each parser:
```yaml
"""AEL"""
'HDFS': {
  'minEventCount': 2,
  'merge_percent': 0.5
}
'BGL': {
  'minEventCount': 2,
  'merge_percent': 0.5
}
'Thunderbird': {
  'minEventCount': 2,
  'merge_percent': 0.4
}
'Spirit': {
  'minEventCount': 2,
  'merge_percent': 0.4
}

"""Spell"""
'HDFS': {
  'tau': 0.7
}
'BGL': {
  'tau': 0.75
}
'Thunderbird': {
  'tau': 0.5
}
'Spirit': {
  'tau': 0.5
}

"""Drain"""
'HDFS': {
  'st': 0.5,
  'depth': 4
}
'BGL': {
  'st': 0.5,
  'depth': 4
}
'Thunderbird': {
  'st': 0.5,
  'depth': 4
}
'Spirit': {
  'st': 0.5,
  'depth': 4
}

"""IPLoM"""
'HDFS': {
  'CT': 0.35,
  'lowerBound': 0.25
}
'BGL': {
  'CT': 0.4,
  'lowerBound': 0.01
}
'Thunderbird': {
  'CT': 0.3,
  'lowerBound': 0.2
}
'Spirit': {
  'CT': 0.3,
  'lowerBound': 0.2
}

```
