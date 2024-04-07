var npsVideos = [
    ["377_sample01_rep00_pose.mp4", "377_sample01_rep00_hn.mp4", "377_sample01_rep00_mh.mp4", "377_sample01_rep00_ours.mp4"],
    ["387_sample02_rep00_pose.mp4", "387_sample02_rep00_hn.mp4", "387_sample02_rep00_mh.mp4", "387_sample02_rep00_ours.mp4"],
    ["392_sample07_rep00_pose.mp4", "392_sample07_rep00_hn.mp4", "392_sample07_rep00_mh.mp4", "392_sample07_rep00_ours.mp4"],
    ["393_sample06_rep00_pose.mp4", "393_sample06_rep00_hn.mp4", "393_sample06_rep00_mh.mp4", "393_sample06_rep00_ours.mp4"],
    ["394_sample05_rep00_pose.mp4", "394_sample05_rep00_hn.mp4", "394_sample05_rep00_mh.mp4", "394_sample05_rep00_ours.mp4"],
    ["wildvideo0_sample00_rep00_pose.mp4", "wildvideo0_sample00_rep00_hn.mp4", "wildvideo0_sample00_rep00_mh.mp4", "wildvideo0_sample00_rep00_ours.mp4"],
    ["wildvideo1_sample00_rep00_pose.mp4", "wildvideo1_sample00_rep00_hn.mp4", "wildvideo1_sample00_rep00_mh.mp4", "wildvideo1_sample00_rep00_ours.mp4"],
    ["wildvideo2_sample00_rep00_pose.mp4", "wildvideo2_sample00_rep00_hn.mp4", "wildvideo2_sample00_rep00_mh.mp4", "wildvideo2_sample00_rep00_ours.mp4"],
]
var npsCaptions = [
    "Moving hands around near face.",
    "A person crosses their arms.",
    "Running forward in a diagonal line.",
    "He is running down then stopped and moved his left hand.",
    "A man crouches down as he walks forward and kicks with his right leg.",
    "A person remained sitting down.",
    "A person remained sitting down.",
    "A person remained sitting down."
]

var nvsVideos = [
    ["377_frame000090_rgb_gt.png", "377_frame000090_rgb_hn.mp4", "377_frame000090_rgb_mh.mp4", "377_frame000090_rgb_ours.mp4", "377_frame000090_normal_gt.mp4", "377_frame000090_normal_hn.mp4", "377_frame000090_normal_mh.mp4", "377_frame000090_normal_ours.mp4"],
    ["386_frame000180_rgb_gt.png", "386_frame000180_rgb_hn.mp4", "386_frame000180_rgb_mh.mp4", "386_frame000180_rgb_ours.mp4", "386_frame000180_normal_gt.mp4", "386_frame000180_normal_hn.mp4", "386_frame000180_normal_mh.mp4", "386_frame000180_normal_ours.mp4"],
    ["387_frame000150_rgb_gt.png", "387_frame000150_rgb_hn.mp4", "387_frame000150_rgb_mh.mp4", "387_frame000150_rgb_ours.mp4", "387_frame000150_normal_gt.mp4", "387_frame000150_normal_hn.mp4", "387_frame000150_normal_mh.mp4", "387_frame000150_normal_ours.mp4"],
    ["392_frame000030_rgb_gt.png", "392_frame000030_rgb_hn.mp4", "392_frame000030_rgb_mh.mp4", "392_frame000030_rgb_ours.mp4", "392_frame000030_normal_gt.mp4", "392_frame000030_normal_hn.mp4", "392_frame000030_normal_mh.mp4", "392_frame000030_normal_ours.mp4"],
    ["393_frame000120_rgb_gt.png", "393_frame000120_rgb_hn.mp4", "393_frame000120_rgb_mh.mp4", "393_frame000120_rgb_ours.mp4", "393_frame000120_normal_gt.mp4", "393_frame000120_normal_hn.mp4", "393_frame000120_normal_mh.mp4", "393_frame000120_normal_ours.mp4"],
    ["394_frame000090_rgb_gt.png", "394_frame000090_rgb_hn.mp4", "394_frame000090_rgb_mh.mp4", "394_frame000090_rgb_ours.mp4", "394_frame000090_normal_gt.mp4", "394_frame000090_normal_hn.mp4", "394_frame000090_normal_mh.mp4", "394_frame000090_normal_ours.mp4"],
]

function ChangeSceneNPS(idx){
    var li_list = document.getElementById("nps-ul").children;
    // var m_list = document.getElementById("method-view-ul").children;
    // console.log(idx);
    // console.log(li_list);

    for(i = 0; i < li_list.length; i++){
        if (li_list[i].className === "disabled"){
            continue
        }
        li_list[i].className = "";
    }
    if (idx < 5) {
        li_list[idx].className = "active";
    } else {
        li_list[idx + 1].className = "active";
    }

    currentVideos = npsVideos[idx]
    document.getElementById("nps_pose").src = "./files/nps_mdm/" + currentVideos[0];
    document.getElementById("nps_hn").src = "./files/nps_mdm/" + currentVideos[1];
    document.getElementById("nps_mh").src = "./files/nps_mdm/" + currentVideos[2];
    document.getElementById("nps_ours").src = "./files/nps_mdm/" + currentVideos[3];
    document.getElementById("nps_caption").innerText = npsCaptions[idx];
}

function ChangeSceneNVS(idx){
    var li_list = document.getElementById("nvs-ul").children;
    for(i = 0; i < li_list.length; i++){
        if (li_list[i].className === "disabled"){
            continue
        }
        li_list[i].className = "";
    }
    li_list[idx].className = "active";
    
    currentVideos = nvsVideos[idx]
    document.getElementById("nvs_rgb_refimg").src = "./files/nvs/" + currentVideos[0];
    document.getElementById("nvs_rgb_hn").src = "./files/nvs/" + currentVideos[1];
    document.getElementById("nvs_rgb_mh").src = "./files/nvs/" + currentVideos[2];
    document.getElementById("nvs_rgb_ours").src = "./files/nvs/" + currentVideos[3];
    document.getElementById("nvs_normal_gt").src = "./files/nvs/" + currentVideos[4];
    document.getElementById("nvs_normal_hn").src = "./files/nvs/" + currentVideos[5];
    document.getElementById("nvs_normal_mh").src = "./files/nvs/" + currentVideos[6];
    document.getElementById("nvs_normal_ours").src = "./files/nvs/" + currentVideos[7];
}