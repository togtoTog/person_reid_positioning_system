
$(document).ready(function () {
    setInterval(function () {
        function loadImages(data) {
            var box = document.getElementById('imgs');
            box.innerHTML = "";
            imgs = data.split('.jpg') 
            danger_nums = imgs[imgs.length - 1].split(',')
            for (var i = imgs.length - 2; i >=0 ; i--) {
                
                var oImg = new Image();
                oImg.src = `static/img/danger/` + imgs[i] + `.jpg`;

                oImg.onclick = function () {
                    // var tag = document.getElementById('tag')
                    // tag.innerHTML = '已选中图片'
                    // tag.style.color = 'red';
                    this.style.padding = '5px 5px';
                    this.style.marginTop = '10px';
                    this.style.backgroundColor = 'rgb(247, 11, 11)';
                    var img = document.getElementById("img");
                    //设置属性和src
                    img.src = this.src;
                }
                oImg.style = "margin-bottom: 10px;"
                oTime = danger_nums[i]/10
                var oText = new Text(`接触${oTime}秒`)
                
                
                var oDiv = document.createElement("div");
                oDiv.style = "text-align: right;font-size: 20px;margin-left: 20px;"
                oDiv.style.width = "120px"
                oDiv.style.heigth = "400px"
                oDiv.style.float = "left"
                

                oDiv.appendChild(oImg)
                oDiv.appendChild(oText);
                box.appendChild(oDiv);
            }
            // var pic = document.getElementById('picture_v2');
            // var next = document.querySelector(".arrow_right");
            // var prev = document.querySelector(".arrow_left");
            // var imgwidth = pic.children[0].offsetWidth;
            // var margin_left = pic.children[0].offsetLeft;
            // imgwidth += margin_left;
            // var move = 0;
            // next.onclick = function () {
            //     if (move == pic.children.length - 1) {
            //         move = 0;
            //         pic.style.left = 0 + "px";
            //     }
            //     move++;
            //     animate(pic, -move * imgwidth);
            // }
            // prev.onclick = function () {
            //     if (move == 0) {
            //         move = pic.children.length - 1;
            //         pic.style.left = -move * imgwidth + "px";
            //     }
            //     move--;
            //     animate(pic, -move * imgwidth);
            // }
            var timer = null;
            function autoPlay() {
                timer = setInterval(function () {
                    next.onclick();
                }, 2000);
            }
            // autoPlay();
            // var Carousel = document.querySelector(".Carousel");
            // Carousel.onmouseenter = function () {
            //     clearInterval(timer);
            // }
            // Carousel.onmouseleave = function () {
            //     autoPlay();
            // }
            function animate(element, distance) {
                clearInterval(element.timer)
                element.timer = setInterval(function () {
                    var present = element.offsetLeft;//获取元素的当前的位置
                    var movement = 10;//每次移动的距离
                    movement = present < distance ? movement : -movement;
                    present += movement;//当前移动到位置
                    if (Math.abs(present - distance) > Math.abs(movement)) {
                        element.style.left = present + "px"
                    } else {
                        clearInterval(element.timer);
                        element.style.left = distance + "px"
                    }
                }, 10);
            }
        }
        $.ajax({
            type: "GET", // 数据提交类型
            url: "/getlen", // 发送地址
            // data: formData, //发送数据
            async: true, // 是否异步
            processData: false, //processData 默认为false，当设置为true的时候,jquery ajax 提交的时候不会序列化 data，而是直接使用data
            contentType: false,
            success: function (data) {
                // if (data.code === 200) {
                //     console.log(`${data.message}`);
                // } else {
                //     console.log(`${data.message}`);
                // }
                console.log(data)
                // 定义一个img
                loadImages(data)
                //将图片添加到页面中
                // document.body.appendChild(img);
                // document.getElementById(block).appendChild(img);
                // console.log(data.license_plate)
                // $("#license_plate").html(data);
            },
            error: function (e) {
                self.$message.warning(`${e}`);
                //console.log("不成功"+e);
            }
        });
        // stream.getTracks()[0].stop();//结束关闭流
    }, 300);
});