// 全局变量
let currentPage = 1;
const itemsPerPage = 10;
let currentResults = [];
let currentModel = 'resnet50';
let currentAugmentationType = 'rotation';
let currentParameters = {};

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
  console.log('页面加载完成，开始初始化...');
  try {
    // 初始化页面
    initPage();
    
    // 绑定事件监听器
    bindEventListeners();
    console.log('页面初始化完成');
  } catch (error) {
    console.error('页面初始化失败:', error);
  }
});

// 初始化页面
function initPage() {
  console.log('初始化页面...');
  // 加载模型选项
  loadModelOptions();
  
  // 加载数据增强类型
  loadAugmentationTypes();
  
  // 加载默认结果
  loadResults();
}

// 绑定事件监听器
function bindEventListeners() {
  console.log('绑定事件监听器...');
  // 运行评估按钮
  const runBtn = document.getElementById('runEvaluationBtn');
  if (runBtn) {
    runBtn.addEventListener('click', runEvaluation);
    console.log('运行评估按钮事件绑定成功');
  }
  
  // 加载结果按钮
  const loadBtn = document.getElementById('loadResultsBtn');
  if (loadBtn) {
    loadBtn.addEventListener('click', function() {
      console.log('加载结果按钮点击');
      // 获取当前选择的参数
      const modelSelect = document.getElementById('modelSelect');
      const augmentationTypeSelect = document.getElementById('augmentationType');
      
      if (!modelSelect || !augmentationTypeSelect) {
        console.error('模型选择或数据增强类型选择元素不存在');
        return;
      }
      
      const model = modelSelect.value;
      const augmentationType = augmentationTypeSelect.value;
      
      // 收集参数
      let parameters = {};
      if (augmentationType === 'rotation') {
        const rotationAngleSelect = document.getElementById('rotationAngle');
        if (rotationAngleSelect) {
          parameters.rotation_angle = rotationAngleSelect.value;
        }
      } else if (augmentationType === 'cutting') {
        const cropTypeSelect = document.getElementById('cropType');
        const cropRatioSelect = document.getElementById('cropRatio');
        if (cropTypeSelect) {
          parameters.crop_type = cropTypeSelect.value;
        }
        if (cropRatioSelect) {
          parameters.crop_ratio = cropRatioSelect.value;
        }
        // 添加偏移百分比参数
        const offsetXSelect = document.getElementById('offsetXPercent');
        const offsetYSelect = document.getElementById('offsetYPercent');
        if (offsetXSelect) {
          parameters.offset_x_percent = offsetXSelect.value;
        }
        if (offsetYSelect) {
          parameters.offset_y_percent = offsetYSelect.value;
        }
      } else if (augmentationType === 'rotation_cutting') {
        // 添加旋转角度参数
        const rotationAngleSelect = document.getElementById('rotationAngle');
        if (rotationAngleSelect) {
          parameters.rotation_angle = rotationAngleSelect.value;
        }
        // 添加裁剪参数
        const cropTypeSelect = document.getElementById('cropType');
        const cropRatioSelect = document.getElementById('cropRatio');
        if (cropTypeSelect) {
          parameters.crop_type = cropTypeSelect.value;
        }
        if (cropRatioSelect) {
          parameters.crop_ratio = cropRatioSelect.value;
        }
        // 添加偏移百分比参数
        const offsetXSelect = document.getElementById('offsetXPercent');
        const offsetYSelect = document.getElementById('offsetYPercent');
        if (offsetXSelect) {
          parameters.offset_x_percent = offsetXSelect.value;
        }
        if (offsetYSelect) {
          parameters.offset_y_percent = offsetYSelect.value;
        }
      }
      
      // 更新全局变量
      currentModel = model;
      currentAugmentationType = augmentationType;
      currentParameters = parameters;
      currentPage = 1;
      
      console.log('加载结果，参数:', { model, augmentationType, parameters });
      // 加载结果
      loadResults();
    });
    console.log('加载结果按钮事件绑定成功');
  }
  
  // 数据增强类型变化
  const augmentationTypeSelect = document.getElementById('augmentationType');
  if (augmentationTypeSelect) {
    augmentationTypeSelect.addEventListener('change', function() {
      console.log('数据增强类型变化:', this.value);
      updateAugmentationParameters();
    });
    console.log('数据增强类型变化事件绑定成功');
  }
  
  // 帮助按钮点击事件
  const helpBtn = document.getElementById('helpBtn');
  if (helpBtn) {
    helpBtn.addEventListener('click', function(e) {
      e.preventDefault();
      console.log('帮助按钮点击');
      // 暂时禁用模态框功能，避免依赖Bootstrap
      alert('帮助功能暂时不可用，请查看页面上的说明文字。');
    });
    console.log('帮助按钮事件绑定成功');
  }
  
  // 关于按钮点击事件
  const aboutBtn = document.getElementById('aboutBtn');
  if (aboutBtn) {
    aboutBtn.addEventListener('click', function(e) {
      e.preventDefault();
      console.log('关于按钮点击');
      // 暂时禁用模态框功能，避免依赖Bootstrap
      alert('关于功能暂时不可用，请查看页面上的说明文字。');
    });
    console.log('关于按钮事件绑定成功');
  }
  
  // 分页按钮
  const pagination = document.getElementById('pagination');
  if (pagination) {
    pagination.addEventListener('click', function(e) {
      if (e.target.tagName === 'A' && e.target.dataset.page) {
        e.preventDefault();
        currentPage = parseInt(e.target.dataset.page);
        console.log('点击分页按钮，跳转到第', currentPage, '页');
        // 重新加载对应页面的数据
        loadResults();
      }
    });
  }
}

// 加载模型选项
function loadModelOptions() {
  console.log('加载模型选项...');
  fetch('/api/models')
    .then(response => response.json())
    .then(data => {
      console.log('模型选项加载成功:', data);
      const modelSelect = document.getElementById('modelSelect');
      if (modelSelect) {
        modelSelect.innerHTML = '';
        
        const models = data.models || [];
        models.forEach(model => {
          const option = document.createElement('option');
          option.value = model;
          option.textContent = model;
          modelSelect.appendChild(option);
        });
        
        // 设置默认模型
        if (models.length > 0) {
          currentModel = models[0];
          modelSelect.value = currentModel;
        }
      } else {
        console.error('模型选择元素不存在');
      }
    })
    .catch(error => {
      console.error('Error loading models:', error);
      showAlert('加载模型失败', 'danger');
    });
}

// 加载数据增强类型
function loadAugmentationTypes() {
  console.log('加载数据增强类型...');
  fetch('/api/augmentation/types')
    .then(response => response.json())
    .then(data => {
      console.log('数据增强类型加载成功:', data);
      const typeSelect = document.getElementById('augmentationType');
      if (typeSelect) {
        typeSelect.innerHTML = '';
        
        const types = data.augmentation_types || [];
        types.forEach(type => {
          const option = document.createElement('option');
          option.value = type;
          let displayText = '裁剪';
          if (type === 'rotation') {
            displayText = '旋转';
          } else if (type === 'rotation_cutting') {
            displayText = '旋转+裁剪';
          }
          option.textContent = displayText;
          typeSelect.appendChild(option);
        });
        
        // 设置默认类型
        if (types.length > 0) {
          currentAugmentationType = types[0];
          typeSelect.value = currentAugmentationType;
          updateAugmentationParameters();
        }
      } else {
        console.error('数据增强类型选择元素不存在');
      }
    })
    .catch(error => {
      console.error('Error loading augmentation types:', error);
      showAlert('加载数据增强类型失败', 'danger');
    });
}

// 更新数据增强参数选项
function updateAugmentationParameters() {
  const type = document.getElementById('augmentationType').value;
  currentAugmentationType = type;
  console.log('更新数据增强参数选项，类型:', type);
  
  // 显示/隐藏相应的参数分组
  const rotationGroup = document.getElementById('rotationAngleGroup');
  const cropTypeGroup = document.getElementById('cropTypeGroup');
  const cropRatioGroup = document.getElementById('cropRatioGroup');
  const offsetXGroup = document.getElementById('offsetXGroup');
  const offsetYGroup = document.getElementById('offsetYGroup');
  
  if (rotationGroup && cropTypeGroup && cropRatioGroup) {
    if (type === 'rotation') {
      rotationGroup.style.display = 'block';
      cropTypeGroup.style.display = 'none';
      cropRatioGroup.style.display = 'none';
      if (offsetXGroup) offsetXGroup.style.display = 'none';
      if (offsetYGroup) offsetYGroup.style.display = 'none';
    } else if (type === 'cutting') {
      rotationGroup.style.display = 'none';
      cropTypeGroup.style.display = 'block';
      cropRatioGroup.style.display = 'block';
      // 默认隐藏偏移百分比选项，只有选择offset类型时才显示
      if (offsetXGroup) offsetXGroup.style.display = 'none';
      if (offsetYGroup) offsetYGroup.style.display = 'none';
    } else if (type === 'rotation_cutting') {
      rotationGroup.style.display = 'block';
      cropTypeGroup.style.display = 'block';
      cropRatioGroup.style.display = 'block';
      // 默认隐藏偏移百分比选项，只有选择offset类型时才显示
      if (offsetXGroup) offsetXGroup.style.display = 'none';
      if (offsetYGroup) offsetYGroup.style.display = 'none';
    }
  }
  
  // 获取参数选项
  fetch(`/api/augmentation/parameters?type=${type}`)
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log('参数选项加载成功:', data);
      
      // 首先处理旋转角度选项
      if (type === 'rotation' || type === 'rotation_cutting') {
        // 更新旋转角度选项
        const rotationAngleSelect = document.getElementById('rotationAngle');
        if (rotationAngleSelect) {
          rotationAngleSelect.innerHTML = '<option value="0">全部角度</option>';
          
          if (data.rotation_angles && data.rotation_angles.length > 0) {
            console.log('加载旋转角度选项:', data.rotation_angles);
            data.rotation_angles.forEach(angle => {
              const option = document.createElement('option');
              option.value = angle;
              option.textContent = `${angle}度`;
              rotationAngleSelect.appendChild(option);
            });
          } else {
            console.error('没有获取到旋转角度选项');
            // 如果没有获取到旋转角度选项，尝试加载默认的角度选项
            const defaultAngles = [15, 30, 45, 345];
            defaultAngles.forEach(angle => {
              const option = document.createElement('option');
              option.value = angle;
              option.textContent = `${angle}度`;
              rotationAngleSelect.appendChild(option);
            });
          }
        }
      }
      
      // 处理裁剪类型和比例选项
      if (type === 'cutting' || type === 'rotation_cutting') {
        // 更新裁剪类型选项
        const cropTypeSelect = document.getElementById('cropType');
        if (cropTypeSelect) {
          cropTypeSelect.innerHTML = '<option value="">全部类型</option>';
          
          if (data.crop_types && data.crop_types.length > 0) {
            console.log('加载裁剪类型选项:', data.crop_types);
            data.crop_types.forEach(cropType => {
              const option = document.createElement('option');
              option.value = cropType;
              option.textContent = cropType;
              cropTypeSelect.appendChild(option);
            });
          } else {
            console.error('没有获取到裁剪类型选项');
            // 如果没有获取到裁剪类型选项，尝试加载默认的类型选项
            const defaultTypes = ['center', 'offset'];
            defaultTypes.forEach(type => {
              const option = document.createElement('option');
              option.value = type;
              option.textContent = type;
              cropTypeSelect.appendChild(option);
            });
          }
        }
        
        // 更新裁剪比例选项
        const cropRatioSelect = document.getElementById('cropRatio');
        if (cropRatioSelect) {
          cropRatioSelect.innerHTML = '<option value="0.0">全部比例</option>';
          
          if (data.crop_ratios && data.crop_ratios.length > 0) {
            console.log('加载裁剪比例选项:', data.crop_ratios);
            data.crop_ratios.forEach(ratio => {
              const option = document.createElement('option');
              option.value = ratio;
              option.textContent = ratio;
              cropRatioSelect.appendChild(option);
            });
          } else {
            console.error('没有获取到裁剪比例选项');
            // 如果没有获取到裁剪比例选项，尝试加载默认的比例选项
            const defaultRatios = [0.7, 0.8, 0.9];
            defaultRatios.forEach(ratio => {
              const option = document.createElement('option');
              option.value = ratio;
              option.textContent = ratio;
              cropRatioSelect.appendChild(option);
            });
          }
        }
        
        // 更新偏移X百分比选项
        const offsetXSelect = document.getElementById('offsetXPercent');
        if (offsetXSelect) {
          offsetXSelect.innerHTML = '<option value="0">全部X偏移</option>';
          
          if (data.offset_x_percents && data.offset_x_percents.length > 0) {
            console.log('加载偏移X百分比选项:', data.offset_x_percents);
            data.offset_x_percents.forEach(offset => {
              const option = document.createElement('option');
              option.value = offset;
              option.textContent = offset + '%';
              offsetXSelect.appendChild(option);
            });
          } else {
            console.error('没有获取到偏移X百分比选项');
            // 如果没有获取到偏移X百分比选项，尝试加载默认的偏移选项
            const defaultOffsets = [-15, -10, 10, 15];
            defaultOffsets.forEach(offset => {
              const option = document.createElement('option');
              option.value = offset;
              option.textContent = offset + '%';
              offsetXSelect.appendChild(option);
            });
          }
        }
        
        // 更新偏移Y百分比选项
        const offsetYSelect = document.getElementById('offsetYPercent');
        if (offsetYSelect) {
          offsetYSelect.innerHTML = '<option value="0">全部Y偏移</option>';
          
          if (data.offset_y_percents && data.offset_y_percents.length > 0) {
            console.log('加载偏移Y百分比选项:', data.offset_y_percents);
            data.offset_y_percents.forEach(offset => {
              const option = document.createElement('option');
              option.value = offset;
              option.textContent = offset + '%';
              offsetYSelect.appendChild(option);
            });
          } else {
            console.error('没有获取到偏移Y百分比选项');
            // 如果没有获取到偏移Y百分比选项，尝试加载默认的偏移选项
            const defaultOffsets = [-15, -10, 10, 15];
            defaultOffsets.forEach(offset => {
              const option = document.createElement('option');
              option.value = offset;
              option.textContent = offset + '%';
              offsetYSelect.appendChild(option);
            });
          }
        }
      }
      
      // 对于旋转+裁剪类型，添加事件监听
      if (type === 'rotation_cutting') {
        const rotationAngleSelect = document.getElementById('rotationAngle');
        const cropTypeSelect = document.getElementById('cropType');
        
        // 添加旋转角度变化事件监听
        if (rotationAngleSelect) {
          rotationAngleSelect.onchange = function() {
            const selectedAngle = this.value;
            console.log('选择的旋转角度:', selectedAngle);
            
            // 重新加载裁剪类型和比例选项，基于当前选择的角度
            const type = document.getElementById('augmentationType').value;
            const cropTypeSelect = document.getElementById('cropType');
            const selectedCropType = cropTypeSelect ? cropTypeSelect.value : '';
            
            console.log('发送请求:', `/api/augmentation/parameters?type=${type}&crop_type=${selectedCropType}&rotation_angle=${selectedAngle}`);
            
            fetch(`/api/augmentation/parameters?type=${type}&crop_type=${selectedCropType}&rotation_angle=${selectedAngle}`)
              .then(response => response.json())
              .then(data => {
                console.log('基于旋转角度的参数加载成功:', data);
                
                // 更新裁剪类型选项
                if (cropTypeSelect) {
                  cropTypeSelect.innerHTML = '<option value="">全部类型</option>';
                  
                  if (data.crop_types && data.crop_types.length > 0) {
                    data.crop_types.forEach(cropType => {
                      const option = document.createElement('option');
                      option.value = cropType;
                      option.textContent = cropType;
                      cropTypeSelect.appendChild(option);
                    });
                  }
                }
                
                // 更新裁剪比例选项
                const cropRatioSelect = document.getElementById('cropRatio');
                if (cropRatioSelect) {
                  cropRatioSelect.innerHTML = '<option value="0.0">全部比例</option>';
                  
                  if (data.crop_ratios && data.crop_ratios.length > 0) {
                    data.crop_ratios.forEach(ratio => {
                      const option = document.createElement('option');
                      option.value = ratio;
                      option.textContent = ratio;
                      cropRatioSelect.appendChild(option);
                    });
                  }
                }
              })
              .catch(error => {
                console.error('Error loading parameters based on rotation angle:', error);
              });
          };
          
          // 触发初始的旋转角度变化事件
          setTimeout(() => {
            if (rotationAngleSelect.onchange) {
              rotationAngleSelect.onchange();
            }
          }, 100);
        }
        
        // 添加裁剪类型变化事件监听
        if (cropTypeSelect) {
          cropTypeSelect.onchange = function() {
            const selectedType = this.value;
            console.log('选择的裁剪类型:', selectedType);
            
            // 获取最新的元素引用
            const offsetXGroup = document.getElementById('offsetXGroup');
            const offsetYGroup = document.getElementById('offsetYGroup');
            
            // 根据选择的裁剪类型显示/隐藏偏移百分比选项
            if (selectedType === 'offset') {
              if (offsetXGroup) offsetXGroup.style.display = 'block';
              if (offsetYGroup) offsetYGroup.style.display = 'block';
            } else {
              if (offsetXGroup) offsetXGroup.style.display = 'none';
              if (offsetYGroup) offsetYGroup.style.display = 'none';
            }
            
            // 重新加载裁剪比例选项，基于当前选择的裁剪类型和旋转角度
            const type = document.getElementById('augmentationType').value;
            const rotationAngleSelect = document.getElementById('rotationAngle');
            const selectedAngle = rotationAngleSelect ? rotationAngleSelect.value : 0;
            
            console.log('发送请求:', `/api/augmentation/parameters?type=${type}&crop_type=${selectedType}&rotation_angle=${selectedAngle}`);
            
            fetch(`/api/augmentation/parameters?type=${type}&crop_type=${selectedType}&rotation_angle=${selectedAngle}`)
              .then(response => response.json())
              .then(data => {
                console.log('基于裁剪类型和旋转角度的参数加载成功:', data);
                
                // 更新裁剪比例选项
                const cropRatioSelect = document.getElementById('cropRatio');
                if (cropRatioSelect) {
                  cropRatioSelect.innerHTML = '<option value="0.0">全部比例</option>';
                  
                  if (data.crop_ratios && data.crop_ratios.length > 0) {
                    data.crop_ratios.forEach(ratio => {
                      const option = document.createElement('option');
                      option.value = ratio;
                      option.textContent = ratio;
                      cropRatioSelect.appendChild(option);
                    });
                  }
                }
              })
              .catch(error => {
                console.error('Error loading crop ratio parameters:', error);
              });
          };
        }
        
        // 添加裁剪比例变化事件监听
        const cropRatioSelect = document.getElementById('cropRatio');
        if (cropRatioSelect) {
          cropRatioSelect.onchange = function() {
            const selectedRatio = this.value;
            console.log('选择的裁剪比例:', selectedRatio);
            
            // 加载该裁剪比例下的偏移百分比选项
            if (selectedRatio) {
              const type = document.getElementById('augmentationType').value;
              const rotationAngleSelect = document.getElementById('rotationAngle');
              const selectedAngle = rotationAngleSelect ? rotationAngleSelect.value : 0;
              const cropTypeSelect = document.getElementById('cropType');
              const selectedCropType = cropTypeSelect ? cropTypeSelect.value : '';
              
              fetch(`/api/augmentation/parameters?type=${type}&crop_type=${selectedCropType}&rotation_angle=${selectedAngle}&crop_ratio=${selectedRatio}`)
                .then(response => response.json())
                .then(data => {
                  console.log('偏移百分比选项加载成功:', data);
                  
                  // 更新偏移X百分比选项
                  const offsetXSelect = document.getElementById('offsetXPercent');
                  if (offsetXSelect) {
                    offsetXSelect.innerHTML = '<option value="0">全部X偏移</option>';
                    
                    if (data.offset_x_percents && data.offset_x_percents.length > 0) {
                      data.offset_x_percents.forEach(offset => {
                        const option = document.createElement('option');
                        option.value = offset;
                        option.textContent = offset + '%';
                        offsetXSelect.appendChild(option);
                      });
                    }
                  }
                  
                  // 更新偏移Y百分比选项
                  const offsetYSelect = document.getElementById('offsetYPercent');
                  if (offsetYSelect) {
                    offsetYSelect.innerHTML = '<option value="0">全部Y偏移</option>';
                    
                    if (data.offset_y_percents && data.offset_y_percents.length > 0) {
                      data.offset_y_percents.forEach(offset => {
                        const option = document.createElement('option');
                        option.value = offset;
                        option.textContent = offset + '%';
                        offsetYSelect.appendChild(option);
                      });
                    }
                  }
                })
                .catch(error => {
                  console.error('Error loading offset parameters:', error);
                });
            }
          };
        }
      }
      
      // 对于裁剪类型，添加事件监听
      if (type === 'cutting') {
        const cropTypeSelect = document.getElementById('cropType');
        const cropRatioSelect = document.getElementById('cropRatio');
        
        // 添加裁剪类型变化事件监听
        if (cropTypeSelect) {
          cropTypeSelect.onchange = function() {
            const selectedType = this.value;
            console.log('选择的裁剪类型:', selectedType);
            
            // 获取最新的元素引用
            const offsetXGroup = document.getElementById('offsetXGroup');
            const offsetYGroup = document.getElementById('offsetYGroup');
            
            // 根据选择的裁剪类型显示/隐藏偏移百分比选项
            if (selectedType === 'offset') {
              if (offsetXGroup) offsetXGroup.style.display = 'block';
              if (offsetYGroup) offsetYGroup.style.display = 'block';
            } else {
              if (offsetXGroup) offsetXGroup.style.display = 'none';
              if (offsetYGroup) offsetYGroup.style.display = 'none';
            }
            
            // 重新加载裁剪比例选项，基于当前选择的裁剪类型
            const type = document.getElementById('augmentationType').value;
            fetch(`/api/augmentation/parameters?type=${type}&crop_type=${selectedType}`)
              .then(response => response.json())
              .then(data => {
                console.log('基于裁剪类型的参数加载成功:', data);
                
                // 更新裁剪比例选项
                const cropRatioSelect = document.getElementById('cropRatio');
                if (cropRatioSelect) {
                  cropRatioSelect.innerHTML = '<option value="0.0">全部比例</option>';
                  
                  if (data.crop_ratios && data.crop_ratios.length > 0) {
                    data.crop_ratios.forEach(ratio => {
                      const option = document.createElement('option');
                      option.value = ratio;
                      option.textContent = ratio;
                      cropRatioSelect.appendChild(option);
                    });
                  }
                }
              })
              .catch(error => {
                console.error('Error loading crop ratio parameters:', error);
              });
          };
        }
        
        // 添加裁剪比例变化事件监听
        if (cropRatioSelect) {
          cropRatioSelect.onchange = function() {
            const selectedRatio = this.value;
            console.log('选择的裁剪比例:', selectedRatio);
            
            // 加载该裁剪比例下的偏移百分比选项
            if (selectedRatio) {
              const type = document.getElementById('augmentationType').value;
              const cropTypeSelect = document.getElementById('cropType');
              const selectedCropType = cropTypeSelect ? cropTypeSelect.value : '';
              
              fetch(`/api/augmentation/parameters?type=${type}&crop_type=${selectedCropType}&crop_ratio=${selectedRatio}`)
                .then(response => response.json())
                .then(data => {
                  console.log('偏移百分比选项加载成功:', data);
                  
                  // 更新偏移X百分比选项
                  const offsetXSelect = document.getElementById('offsetXPercent');
                  if (offsetXSelect) {
                    offsetXSelect.innerHTML = '<option value="0">全部X偏移</option>';
                    
                    if (data.offset_x_percents && data.offset_x_percents.length > 0) {
                      data.offset_x_percents.forEach(offset => {
                        const option = document.createElement('option');
                        option.value = offset;
                        option.textContent = offset + '%';
                        offsetXSelect.appendChild(option);
                      });
                    }
                  }
                  
                  // 更新偏移Y百分比选项
                  const offsetYSelect = document.getElementById('offsetYPercent');
                  if (offsetYSelect) {
                    offsetYSelect.innerHTML = '<option value="0">全部Y偏移</option>';
                    
                    if (data.offset_y_percents && data.offset_y_percents.length > 0) {
                      data.offset_y_percents.forEach(offset => {
                        const option = document.createElement('option');
                        option.value = offset;
                        option.textContent = offset + '%';
                        offsetYSelect.appendChild(option);
                      });
                    }
                  }
                })
                .catch(error => {
                  console.error('Error loading offset parameters:', error);
                });
            }
          };
        }
      }
    })
    .catch(error => {
      console.error('Error loading augmentation parameters:', error);
      showAlert('加载数据增强参数失败', 'danger');
    });
}

// 运行评估
function runEvaluation() {
  const modelSelect = document.getElementById('modelSelect');
  const augmentationTypeSelect = document.getElementById('augmentationType');
  
  if (!modelSelect || !augmentationTypeSelect) {
    showAlert('请选择模型和数据增强类型', 'danger');
    return;
  }
  
  const model = modelSelect.value;
  const augmentationType = augmentationTypeSelect.value;
  let parameters = {};
  
  // 收集参数
  if (augmentationType === 'rotation') {
        const rotationAngleSelect = document.getElementById('rotationAngle');
        if (rotationAngleSelect) {
          parameters.rotation_angle = rotationAngleSelect.value;
        }
      } else if (augmentationType === 'cutting' || augmentationType === 'rotation_cutting') {
        const cropTypeSelect = document.getElementById('cropType');
        const cropRatioSelect = document.getElementById('cropRatio');
        const offsetXSelect = document.getElementById('offsetXPercent');
        const offsetYSelect = document.getElementById('offsetYPercent');
        if (cropTypeSelect) {
          parameters.crop_type = cropTypeSelect.value;
        }
        if (cropRatioSelect) {
          parameters.crop_ratio = cropRatioSelect.value;
        }
        if (offsetXSelect) {
          // 确保传递的是纯数字，去除可能的%符号
          parameters.offset_x_percent = parseInt(offsetXSelect.value.toString().replace('%', '')) || 0;
        }
        if (offsetYSelect) {
          // 确保传递的是纯数字，去除可能的%符号
          parameters.offset_y_percent = parseInt(offsetYSelect.value.toString().replace('%', '')) || 0;
        }
      }
      
      // 对于旋转+裁剪类型，还需要添加旋转角度参数
      if (augmentationType === 'rotation_cutting') {
        const rotationAngleSelect = document.getElementById('rotationAngle');
        if (rotationAngleSelect) {
          parameters.rotation_angle = rotationAngleSelect.value;
        }
      }
  
  console.log('运行评估，参数:', { model, augmentationType, parameters });
  
  // 显示评估进度模态框
  $('#runEvaluationModal').modal('show');
  const progressElement = document.getElementById('evaluationLog');
  if (progressElement) {
    progressElement.textContent = '正在运行评估...';
  }
  
  // 发送评估请求
  fetch('/api/run-evaluation', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: model,
      augmentation_type: augmentationType,
      parameters: parameters
    })
  })
  .then(response => response.json())
  .then(data => {
    console.log('评估响应:', data);
    if (data.success) {
      // 评估成功，更新结果
      currentModel = model;
      currentAugmentationType = augmentationType;
      currentParameters = parameters;
      currentPage = 1;
      
      // 加载新的评估结果
      loadResults();
      
      // 关闭模态框
      $('#runEvaluationModal').modal('hide');
      showAlert('评估完成', 'success');
    } else {
      // 评估失败
      if (progressElement) {
        progressElement.textContent = `评估失败: ${data.error || '未知错误'}`;
      }
      setTimeout(() => {
        $('#runEvaluationModal').modal('hide');
      }, 2000);
      showAlert('评估失败', 'danger');
    }
  })
  .catch(error => {
    console.error('Error running evaluation:', error);
    if (progressElement) {
      progressElement.textContent = '评估失败: 网络错误';
    }
    setTimeout(() => {
      $('#runEvaluationModal').modal('hide');
    }, 2000);
    showAlert('评估失败', 'danger');
  });
}

// 加载结果
function loadResults() {
  const params = new URLSearchParams({
    model: currentModel,
    augmentation_type: currentAugmentationType,
    page: currentPage,
    page_size: itemsPerPage
  });
  
  // 添加参数
  if (currentAugmentationType === 'rotation' && currentParameters.rotation_angle) {
      params.append('rotation_angle', currentParameters.rotation_angle);
    } else if (currentAugmentationType === 'cutting') {
      if (currentParameters.crop_type) params.append('crop_type', currentParameters.crop_type);
      if (currentParameters.crop_ratio) params.append('crop_ratio', currentParameters.crop_ratio);
      if (currentParameters.offset_x_percent) {
        // 确保传递的是纯数字，去除可能的%符号
        const offsetX = parseInt(currentParameters.offset_x_percent.toString().replace('%', '')) || 0;
        params.append('offset_x_percent', offsetX);
      }
      if (currentParameters.offset_y_percent) {
        // 确保传递的是纯数字，去除可能的%符号
        const offsetY = parseInt(currentParameters.offset_y_percent.toString().replace('%', '')) || 0;
        params.append('offset_y_percent', offsetY);
      }
    } else if (currentAugmentationType === 'rotation_cutting') {
      if (currentParameters.rotation_angle) params.append('rotation_angle', currentParameters.rotation_angle);
      if (currentParameters.crop_type) params.append('crop_type', currentParameters.crop_type);
      if (currentParameters.crop_ratio) params.append('crop_ratio', currentParameters.crop_ratio);
      if (currentParameters.offset_x_percent) {
        // 确保传递的是纯数字，去除可能的%符号
        const offsetX = parseInt(currentParameters.offset_x_percent.toString().replace('%', '')) || 0;
        params.append('offset_x_percent', offsetX);
      }
      if (currentParameters.offset_y_percent) {
        // 确保传递的是纯数字，去除可能的%符号
        const offsetY = parseInt(currentParameters.offset_y_percent.toString().replace('%', '')) || 0;
        params.append('offset_y_percent', offsetY);
      }
    }
  
  console.log('加载结果，参数:', params.toString());
  
  // 显示加载状态
  showLoading();
  
  fetch(`/api/evaluation/results?${params.toString()}`)
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log('结果加载成功:', data);
      // 检查数据格式
      if (data.success || data.results) {
        // 更新结果
        currentResults = data.results || [];
        
        // 渲染结果
        renderResults();
        
        // 渲染指标
        renderMetrics(data.metrics || {});
        
        // 渲染分页
        const total = data.total || data.pagination?.total_results || 0;
        const page = data.page || data.pagination?.current_page || 1;
        const per_page = data.per_page || data.pagination?.page_size || itemsPerPage;
        renderPagination(total, page, per_page);
      } else if (data.error) {
        console.error('加载结果失败:', data.error || '未知错误');
        showAlert('加载结果失败: ' + (data.error || '未知错误'), 'danger');
        // 渲染空结果
        currentResults = [];
        renderResults();
      } else {
        console.error('加载结果失败: 未知响应格式');
        showAlert('加载结果失败: 未知响应格式', 'danger');
        // 渲染空结果
        currentResults = [];
        renderResults();
      }
    })
    .catch(error => {
      console.error('Error loading results:', error);
      showAlert('加载结果失败: ' + error.message, 'danger');
      // 渲染空结果
      currentResults = [];
      renderResults();
    })
    .finally(() => {
      // 隐藏加载状态
      hideLoading();
      console.log('隐藏加载状态');
    });
}

// 渲染结果
function renderResults() {
  const resultsContainer = document.getElementById('resultsContainer');
  if (!resultsContainer) {
    console.error('结果容器元素不存在');
    return;
  }
  
  resultsContainer.innerHTML = '';
  
  if (currentResults.length === 0) {
    resultsContainer.innerHTML = '<div class="text-center py-4">暂无结果</div>';
    return;
  }
  
  currentResults.forEach((result, index) => {
    const resultItem = document.createElement('div');
    resultItem.className = 'result-item';
    
    // 构建HTML
    let html = `
      <div class="result-header">
        <h3>查询图片: ${result.query_image ? result.query_image.split('/').pop() : '未知'}</h3>
        <div class="result-meta">
          <span>类别: ${result.category || '未知'}</span>
          <span class="ms-3">排名: ${result.rank || '未找到'}</span>
        </div>
      </div>
      <div class="d-flex flex-wrap">
        <div class="me-5">
          <h4>查询图片</h4>
          <div class="image-item">
            <img src="${result.query_image_url || result.augmented_path}" alt="查询图片" class="lazy-image">
          </div>
        </div>
        <div class="flex-grow-1">
          <h4>Top-${result.top_k ? result.top_k.length : 0} 结果</h4>
          <div class="image-grid">
    `;
    
    // 添加top_k结果
    if (result.top_k) {
      result.top_k.forEach((item, idx) => {
        const isCorrect = item.path === result.original_image;
        html += `
          <div class="image-item ${isCorrect ? 'border-2 border-success' : ''}">
            <img src="${item.url}" alt="结果 ${idx + 1}" class="lazy-image">

          </div>
        `;
      });
    }
    
    html += `
          </div>
        </div>
      </div>
    `;
    
    resultItem.innerHTML = html;
    resultsContainer.appendChild(resultItem);
  });
  
  // 初始化懒加载
  initLazyLoading();
}

// 渲染指标
function renderMetrics(metrics) {
  // 显示指标卡片
  const metricsCard = document.getElementById('metricsCard');
  if (metricsCard) {
    metricsCard.style.display = 'block';
  }
  
  // 更新具体指标值
  if (!metrics) return;
  
  document.getElementById('top1Accuracy').textContent = metrics.top1_accuracy ? metrics.top1_accuracy.toFixed(4) : '0.0000';
  document.getElementById('top5Recall').textContent = metrics.top5_recall ? metrics.top5_recall.toFixed(4) : '0.0000';
  document.getElementById('top10Recall').textContent = metrics.top10_recall ? metrics.top10_recall.toFixed(4) : '0.0000';
  document.getElementById('mrr').textContent = metrics.mrr ? metrics.mrr.toFixed(4) : '0.0000';
  document.getElementById('totalQueries').textContent = metrics.total_queries || 0;
  document.getElementById('metricModel').textContent = currentModel;
}

// 渲染分页
function renderPagination(total, currentPage, pageSize) {
  const paginationContainer = document.getElementById('pagination');
  if (!paginationContainer) {
    console.error('分页容器元素不存在');
    return;
  }
  
  paginationContainer.innerHTML = '';
  
  const totalPages = Math.ceil(total / pageSize);
  if (totalPages <= 1) return;
  
  // 上一页按钮
  const prevLi = document.createElement('li');
  prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
  prevLi.innerHTML = `
    <a class="page-link" href="#" data-page="${currentPage - 1}">
      <span aria-hidden="true">&laquo;</span>
    </a>
  `;
  paginationContainer.appendChild(prevLi);
  
  // 页码按钮
  for (let i = 1; i <= totalPages; i++) {
    const li = document.createElement('li');
    li.className = `page-item ${i === currentPage ? 'active' : ''}`;
    li.innerHTML = `
      <a class="page-link" href="#" data-page="${i}">${i}</a>
    `;
    paginationContainer.appendChild(li);
  }
  
  // 下一页按钮
  const nextLi = document.createElement('li');
  nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
  nextLi.innerHTML = `
    <a class="page-link" href="#" data-page="${currentPage + 1}">
      <span aria-hidden="true">&raquo;</span>
    </a>
  `;
  paginationContainer.appendChild(nextLi);
}

// 显示加载状态
function showLoading() {
  const resultsContainer = document.getElementById('resultsContainer');
  if (resultsContainer) {
    resultsContainer.innerHTML = '<div class="text-center py-8"><div class="spinner-border" role="status"><span class="visually-hidden">加载中...</span></div></div>';
  }
  
  // 显示加载 spinner
  const loadingSpinner = document.getElementById('loadingSpinner');
  if (loadingSpinner) {
    loadingSpinner.style.display = 'inline-block';
  }
}

// 隐藏加载状态
function hideLoading() {
  // 隐藏加载 spinner
  const loadingSpinner = document.getElementById('loadingSpinner');
  if (loadingSpinner) {
    loadingSpinner.style.display = 'none';
  }
  
  // 清除 resultsContainer 中的加载状态
  // 注意：这个函数不应该直接清空 resultsContainer，因为这样会清除已经加载的结果
  // 加载状态的清除应该由 renderResults 函数处理
}

// 初始化懒加载
function initLazyLoading() {
  // 简单的懒加载实现
  const lazyImages = document.querySelectorAll('.lazy-image');
  
  if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const image = entry.target;
          image.src = image.dataset.src || image.src;
          imageObserver.unobserve(image);
        }
      });
    });
    
    lazyImages.forEach(image => {
      imageObserver.observe(image);
    });
  } else {
    // 降级方案
    lazyImages.forEach(image => {
      image.src = image.dataset.src || image.src;
    });
  }
}

// 显示提示信息
function showAlert(message, type = 'info') {
  // 创建提示元素
  const alert = document.createElement('div');
  alert.className = `alert alert-${type} alert-dismissible fade show position-fixed top-20 end-0 z-50`;
  alert.role = 'alert';
  alert.innerHTML = `
    ${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
  `;
  
  // 添加到页面
  document.body.appendChild(alert);
  
  // 3秒后自动关闭
  setTimeout(() => {
    alert.classList.remove('show');
    setTimeout(() => {
      alert.remove();
    }, 500);
  }, 3000);
}
