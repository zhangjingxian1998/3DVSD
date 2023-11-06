# '0' 处代表输出
# 正方向为 front up   right
# 负方向为 back  down left
direction_dict = {
    '0':'next to',
    '1':{
        '0':{
            '0':{
                '0':'front',
                '1':{
                    '0':{
                        '0':{
                            '0':'front up right',
                            '1':'front up left'
                        },
                        '1':{
                            '0':'front down right',
                            '1':'front down left'
                        }
                    },# 3重类
                    '1':{
                        '0':{
                            '0':'front up',
                            '1':'front down'
                        },
                        '1':{
                            '0':'front right',
                            '1':'front left'
                        }
                    }# 2重类
                } #多重方向
            },     #Front
            '1':{
                '0':'back',
                '1':{
                    '0':{
                        '0':{
                            '0':'back up right',
                            '1':'back up left'
                        },
                        '1':{
                            '0':'back down right',
                            '1':'back down left'
                        }
                    },# 3重类
                    '1':{
                        '0':{
                            '0':'back up',
                            '1':'back down'
                        },
                        '1':{
                            '0':'back right',
                            '1':'back left'
                        }
                    }# 2重类
                } #多重方向
            }      #Back
        }, # 代表x方向
        '1':{
            '0':{
                '0':'up',
                '1':{
                    '0':{
                        '0':{
                            '0':'up front right',
                            '1':'up front left'
                        },
                        '1':{
                            '0':'up back right',
                            '1':'up back left'
                        }
                    },# 3重类
                    '1':{
                        '0':{
                            '0':'up front',
                            '1':'up down'
                        },
                        '1':{
                            '0':'up right',
                            '1':'up left'
                        }
                    }# 2重类
                } #多重方向
            },     #UP
            '1':{
                '0':'down',
                '1':{
                    '0':{
                        '0':{
                            '0':'down front right',
                            '1':'down front left'
                        },
                        '1':{
                            '0':'down back right',
                            '1':'down back left'
                        }
                    },# 3重类
                    '1':{
                        '0':{
                            '0':'down front',
                            '1':'down back'
                        },
                        '1':{
                            '0':'down right',
                            '1':'down left'
                        }
                    }# 2重类
                } #多重方向
            }      #Down
        }, # 代表y方向
        '2':{
            '0':{
                '0':'right',
                '1':{
                    '0':{
                        '0':{
                            '0':'right front up',
                            '1':'right front down'
                        },
                        '1':{
                            '0':'right back up',
                            '1':'right back down'
                        }
                    },# 3重类
                    '1':{
                        '0':{
                            '0':'right front',
                            '1':'right back'
                        },
                        '1':{
                            '0':'right up',
                            '1':'right down'
                        }
                    }# 2重类
                } #多重方向
            },     #Right
            '1':{
                '0':'left',
                '1':{
                    '0':{
                        '0':{
                            '0':'left front up',
                            '1':'left front down'
                        },
                        '1':{
                            '0':'left back up',
                            '1':'left back down'
                        }
                    },# 3重类
                    '1':{
                        '0':{
                            '0':'left front',
                            '1':'left back'
                        },
                        '1':{
                            '0':'left up',
                            '1':'left down'
                        }
                    }# 2重类
                } #多重方向
            }      #Left
        }  # 代表z方向
    },
    '2':'void',#没有middle类的情况
    '3':'void'#没有middle类的情况
}
