let video_id = "o9D0gTqfm6c"

let summaryTxt = document.getElementById("summaryText")
let summaryLen = document.getElementById("summaryLength")
let summaryDuration = document.getElementById("summaryTime")
let summaryButton = document.getElementById("summaryBtn")

let API_response = ""
let APIresult = {
    "Summary" : "",
    "summary_length" : "",
    "timeTakenS" : 0
}
async function getSummary(){
    try{
        const res = await axios.get(`http://localhost:5000/formDisplay/getSummary?videoID=${video_id}`)
        console.log(res.data)
        API_response = res.data
        return res.data
    }catch(error){
        console.log("Error is: ",error)
    }
}

summaryButton.addEventListener('click',async()=>{
    console.log("button clicked")
    result =  await getSummary()
    console.log("Result is: ",result)
    console.log("API Response is: ",API_response)
    APIresult.Summary = result.Summary
    APIresult.summary_length = result.summary_length
    APIresult.timeTakenS = result.timeTakenS
    loadDisplay()
})

async function loadDisplay(){
    summaryTxt.innerHTML = `Summary:  ${APIresult.Summary}`
    summaryLen.innerHTML = `Length of transcript: ${APIresult.summary_length}`
    summaryDuration.innerHTML = `Time Taken to generate summary is: ${APIresult.timeTakenS}`
}
loadDisplay()