document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('imageUpload');
    const searchBtn = document.getElementById('searchBtn');
    const uploadedImage = document.getElementById('uploadedImage');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    
    // 图片预览功能
    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedImage.innerHTML = `<img src="${e.target.result}" alt="上传的图片">`;
            };
            reader.readAsDataURL(file);
        }
    });
    
    // 搜索按钮点击事件
    searchBtn.addEventListener('click', function() {
        const file = imageUpload.files[0];
        if (!file) {
            alert('请先选择一张图片');
            return;
        }
        
        // 显示加载状态
        loading.style.display = 'block';
        results.innerHTML = '';
        
        // 创建FormData对象
        const formData = new FormData();
        formData.append('image', file);
        
        // 发送请求到后端API
        fetch('/api/search', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('搜索失败');
            }
            return response.json();
        })
        .then(data => {
            // 隐藏加载状态
            loading.style.display = 'none';
            
            // 显示搜索结果
            if (data.results && data.results.length > 0) {
                data.results.forEach(item => {
                    const resultItem = document.createElement('div');
                    resultItem.className = 'result-item';
                    resultItem.innerHTML = `
                        <img src="${item.url}" alt="相似图片">
                    `;
                    results.appendChild(resultItem);
                });
            } else {
                results.innerHTML = '<p>未找到相似图片</p>';
            }
        })
        .catch(error => {
            // 隐藏加载状态
            loading.style.display = 'none';
            results.innerHTML = `<p>搜索失败: ${error.message}</p>`;
        });
    });
});