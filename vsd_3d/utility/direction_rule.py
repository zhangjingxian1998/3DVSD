# '0' 处代表输出
# 正方向为 front up   right
# 负方向为 back  down left
extral_id_0 = '<extra_id_0>'
extral_id_1  = '<extra_id_1>'
extral_id_2  = '<extra_id_2>'
extral_id_3  = '<extra_id_3>'
extral_id_4  = '<extra_id_4>'
extral_id_5  = '<extra_id_5>'
extral_id_6  = '<extra_id_6>'
extral_id_7  = '<extra_id_7>'
extral_id_8  = '<extra_id_8>'
extral_id_9  = '<extra_id_9>'
extral_id_10 = '<extra_id_10>'
extral_id_11 = '<extra_id_11>'
extral_id_12 = '<extra_id_12>'
extral_id_13 = '<extra_id_13>'
extral_id_14 = '<extra_id_14>'
extral_id_15 = '<extra_id_15>'
extral_id_16 = '<extra_id_16>'
extral_id_17 = '<extra_id_17>'
extral_id_18 = '<extra_id_18>'
extral_id_19 = '<extra_id_19>'
extral_id_20 = '<extra_id_20>'
extral_id_21 = '<extra_id_21>'
extral_id_22 = '<extra_id_22>'
extral_id_23 = '<extra_id_23>'
extral_id_24 = '<extra_id_24>'
extral_id_25 = '<extra_id_25>'
extral_id_26 = '<extra_id_26>'
direction_dict = {
    '0':extral_id_22, # next
    '1':{
        '0':{
            '0':{
                '0':extral_id_12, # front
                '1':{
                    '0':{
                        '0':{
                            '0':extral_id_20, #front up right
                            '1':extral_id_19  #'front up left'
                        },
                        '1':{
                            '0':extral_id_15, # 'front down right'
                            '1':extral_id_14  # 'front down left'
                        }
                    },# 3重类
                    '1':{
                        '0':{
                            '0':extral_id_18, # 'front up'
                            '1':extral_id_13  # 'front down'
                        },
                        '1':{
                            '0':extral_id_17, # 'front right'
                            '1':extral_id_19  # 'front left'
                        }
                    }# 2重类
                } #多重方向
            },     #Front
            '1':{
                '0':extral_id_0, # 'back'
                '1':{
                    '0':{
                        '0':{
                            '0':extral_id_8, # 'back up right'
                            '1':extral_id_7  # 'back up left'
                        },
                        '1':{
                            '0':extral_id_3, # 'back down right'
                            '1':extral_id_2  # 'back down left'
                        }
                    },# 3重类
                    '1':{
                        '0':{
                            '0':extral_id_6, # 'back up'
                            '1':extral_id_1  # 'back down'
                        },
                        '1':{
                            '0':extral_id_5, # 'back right'
                            '1':extral_id_4  # 'back left'
                        }
                    }# 2重类
                } #多重方向
            }      #Back
        }, # 代表x方向
        '1':{
            '0':{
                '0':extral_id_24, # 'up'
                '1':{
                    '0':{
                        '0':{
                            '0':extral_id_20, # 'up front right'
                            '1':extral_id_19  # 'up front left'
                        },
                        '1':{
                            '0':extral_id_8,  # 'up back right'
                            '1':extral_id_7   # 'up back left'
                        }
                    },# 3重类
                    '1':{
                        '0':{
                            '0':extral_id_18, # 'up front'
                            '1':extral_id_6   # 'up back'
                        },
                        '1':{
                            '0':extral_id_26, # 'up right',
                            '1':extral_id_25  # 'up left'
                        }
                    }# 2重类
                } #多重方向
            },     #UP
            '1':{
                '0':extral_id_9, # 'down'
                '1':{
                    '0':{
                        '0':{
                            '0':extral_id_15, # 'down front right'
                            '1':extral_id_14  # 'down front left'
                        },
                        '1':{
                            '0':extral_id_3,  # 'down back right',
                            '1':extral_id_2   # 'down back left'
                        }
                    },# 3重类
                    '1':{
                        '0':{
                            '0':extral_id_13, # 'down front'
                            '1':extral_id_1   # 'down back'
                        },
                        '1':{
                            '0':extral_id_11, # 'down right',
                            '1':extral_id_10  # 'down left'
                        }
                    }# 2重类
                } #多重方向
            }      #Down
        }, # 代表y方向
        '2':{
            '0':{
                '0':extral_id_23, # 'right'
                '1':{
                    '0':{
                        '0':{
                            '0':extral_id_20, # 'right front up'
                            '1':extral_id_15  # 'right front down'
                        },
                        '1':{
                            '0':extral_id_8,  # 'right back up'
                            '1':extral_id_3   # 'right back down'
                        }
                    },# 3重类
                    '1':{
                        '0':{
                            '0':extral_id_17, # 'right front',
                            '1':extral_id_5   # 'right back'
                        },
                        '1':{
                            '0':extral_id_25, # 'right up',
                            '1':extral_id_11  # 'right down'
                        }
                    }# 2重类
                } #多重方向
            },     #Right
            '1':{
                '0':extral_id_21, # 'left'
                '1':{
                    '0':{
                        '0':{
                            '0':extral_id_19, # 'left front up'
                            '1':extral_id_14  # 'left front down'
                        },
                        '1':{
                            '0':extral_id_7,  # 'left back up'
                            '1':extral_id_2   # 'left back down'
                        }
                    },# 3重类
                    '1':{
                        '0':{
                            '0':extral_id_16, # 'left front',
                            '1':extral_id_4   # 'left back'
                        },
                        '1':{
                            '0':extral_id_25, # 'left up',
                            '1':extral_id_10  # 'left down'
                        }
                    }# 2重类
                } #多重方向
            }      #Left
        }  # 代表z方向
    },
    '2':'void',#没有middle类的情况
    '3':'void'#没有middle类的情况
}
